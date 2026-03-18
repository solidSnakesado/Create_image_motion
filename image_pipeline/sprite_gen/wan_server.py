"""
image_pipeline/sprite_gen/wan_server.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    WAN I2V 파이프라인을 REST API로 노출.
    프론트엔드 대시보드(wan_dashboard.html)가 호출.

엔드포인트:
    POST /api/classify        — Stage 1: 이미지 분류 (키프레임 vs 모션)
    POST /api/generate        — WAN 모션 생성 시작 (비동기)
    GET  /api/status/<job_id> — 생성 진행 상태 조회
    GET  /api/videos/<stem>   — 생성된 영상 목록
    POST /api/select_video    — 영상 선택 → 배경 제거 → Lottie 변환
    POST /api/classify_motion — Stage 2: 모션 키프레임 판단
    GET  /api/files/<path>    — 정적 파일 서빙 (영상/이미지/JSON)

사용법:
    cd ~/anim_pipeline && source animVenv/bin/activate
    python -m image_pipeline.sprite_gen.wan_server
"""

from __future__ import annotations

import json
import logging
import os
import glob
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 폴링 요청 로그 억제 (3초마다 /api/status 호출이 터미널을 채우는 것 방지)
logging.getLogger("werkzeug").setLevel(logging.WARNING)
# Gemini API HTTP 요청 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)

# ── 전역 상태 ────────────────────────────────────────────────

_backend = None       # WanBackend 인스턴스 (지연 초기화)
_converter = None     # WanLottieConverter 인스턴스
_jobs: dict = {}      # job_id → 생성 상태

OUTPUT_DIR = os.environ.get(
    "WAN_OUTPUT_DIR",
    os.path.join(os.path.expanduser("~"), "anim_pipeline", "outputs", "wan")
)
DIR_MOTION          = os.path.join(OUTPUT_DIR, "motion")
DIR_KEYFRAME_ONLY   = os.path.join(OUTPUT_DIR, "keyframe_only")
DIR_MOTION_KEYFRAME = os.path.join(OUTPUT_DIR, "motion_keyframe")


def _get_backend():
    global _backend, _converter
    if _backend is None:
        from dotenv import load_dotenv
        load_dotenv()
        from image_pipeline.sprite_gen.wan_backend import WanBackend
        from image_pipeline.sprite_gen.wan_lottie_converter import WanLottieConverter
        _backend = WanBackend(
            api_key=os.environ["GEMINI_API_KEY"],
            comfyui_url=os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
            output_dir=OUTPUT_DIR,
        )
        _converter = WanLottieConverter()
    return _backend, _converter


# ── Stage 1: 이미지 분류 ─────────────────────────────────────

@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Stage 1: 이미지를 분류하여 키프레임/모션 여부 판단."""
    data = request.json or {}
    image_path = data.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"이미지 없음: {image_path}"}), 400

    backend, _ = _get_backend()
    result = backend.mode_classifier.classify(image_path)

    # 기존 영상 확인
    stem = Path(image_path).stem
    pattern = os.path.join(DIR_MOTION, f"{stem}_attempt*.mp4")
    existing_videos = sorted(glob.glob(pattern))
    existing = []
    for v in existing_videos:
        size_mb = os.path.getsize(v) / (1024 * 1024)
        existing.append({
            "path": v,
            "filename": os.path.basename(v),
            "size_mb": round(size_mb, 2),
        })

    # KEYFRAME_ONLY면 키프레임도 즉시 생성
    keyframe_data = None
    if result.processing_mode.value == "keyframe_only":
        from image_pipeline.sprite_gen.wan_keyframe_generator import WanKeyframeGenerator
        kf_gen = WanKeyframeGenerator()
        kf_config = kf_gen.generate(
            suggested_action=result.suggested_action,
            facing_direction=result.facing_direction.value,
        )
        # JSON 저장
        kf_path = os.path.join(DIR_KEYFRAME_ONLY, f"{stem}_keyframe.json")
        kf_config.to_json(kf_path)
        keyframe_data = kf_config.to_dict()
        keyframe_data["json_path"] = kf_path

    return jsonify({
        "processing_mode":  result.processing_mode.value,
        "facing_direction": result.facing_direction.value,
        "has_deformable":   result.has_deformable,
        "is_scene":         result.is_scene,
        "subject_desc":     result.subject_desc,
        "reason":           result.reason,
        "suggested_action": result.suggested_action,
        "existing_videos":  existing,
        "keyframe_config":  keyframe_data,
    })


# ── WAN 모션 생성 (비동기) ────────────────────────────────────

@app.route("/api/generate", methods=["POST"])
def api_generate():
    """WAN 모션 생성 시작. job_id 반환."""
    data = request.json or {}
    image_path = data.get("image_path", "")
    max_retries = int(data.get("max_retries", 7))
    model_name = data.get("model_name", "")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": f"이미지 없음: {image_path}"}), 400

    # 기존 영상 번호 확인 → 다음 번호부터 시작
    stem = Path(image_path).stem
    existing = sorted(glob.glob(os.path.join(DIR_MOTION, f"{stem}_attempt*.mp4")))
    start_attempt = len(existing)  # 기존 10개면 attempt11부터 시작

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "running",
        "image_path": image_path,
        "stem": stem,
        "max_retries": max_retries,
        "model_name": model_name,
        "start_attempt": start_attempt,
        "started_at": time.time(),
        "result": None,
        "error": None,
    }

    def _run():
        try:
            backend, _ = _get_backend()
            import image_pipeline.sprite_gen.wan_backend as wb
            original_retries = wb.MAX_RETRIES
            original_model = wb.WAN_MODEL
            original_offset = getattr(wb, 'ATTEMPT_OFFSET', 0)

            wb.MAX_RETRIES = max_retries
            wb.ATTEMPT_OFFSET = start_attempt
            if model_name:
                wb.WAN_MODEL = model_name
                logger.info(f"[Generate] 모델 오버라이드: {model_name}")
            try:
                result = backend.generate(image_path)
            finally:
                wb.MAX_RETRIES = original_retries
                wb.WAN_MODEL = original_model
                wb.ATTEMPT_OFFSET = original_offset

            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "success": result.success,
                "video_path": result.video_path,
                "attempts": result.attempts,
            }
        except Exception as e:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    return jsonify({
        "job_id": job_id, "status": "running",
        "max_retries": max_retries, "start_attempt": start_attempt,
    })


@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    """생성 진행 상태 조회 + 현재까지 생성된 영상 목록."""
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404

    elapsed = int(time.time() - job["started_at"])

    # 생성 중에도 이미 디스크에 저장된 영상 목록 반환
    stem = job.get("stem", "")
    pattern = os.path.join(DIR_MOTION, f"{stem}_attempt*.mp4")
    videos = []
    for v in sorted(glob.glob(pattern)):
        size_mb = os.path.getsize(v) / (1024 * 1024)
        videos.append({
            "path": v,
            "filename": os.path.basename(v),
            "size_mb": round(size_mb, 2),
        })

    return jsonify({
        "status": job["status"],
        "elapsed_sec": elapsed,
        "result": job["result"],
        "error": job["error"],
        "videos": videos,
    })


# ── 생성된 영상 목록 ─────────────────────────────────────────

@app.route("/api/videos/<stem>", methods=["GET"])
def api_videos(stem):
    """특정 이미지의 생성된 영상 목록."""
    pattern = os.path.join(DIR_MOTION, f"{stem}_attempt*.mp4")
    videos = sorted(glob.glob(pattern))
    result = []
    for v in videos:
        size_mb = os.path.getsize(v) / (1024 * 1024)
        result.append({
            "path": v,
            "filename": os.path.basename(v),
            "size_mb": round(size_mb, 2),
        })
    return jsonify({"stem": stem, "videos": result})


@app.route("/api/videos_all", methods=["GET"])
def api_videos_all():
    """motion/ 폴더 내 모든 mp4 영상 목록."""
    pattern = os.path.join(DIR_MOTION, "*.mp4")
    videos = sorted(glob.glob(pattern))
    result = []
    for v in videos:
        size_mb = os.path.getsize(v) / (1024 * 1024)
        result.append({
            "path": v,
            "filename": os.path.basename(v),
            "size_mb": round(size_mb, 2),
        })
    return jsonify({"total": len(result), "videos": result})


# ── 영상 선택 → 배경 제거 → Lottie 변환 ─────────────────────

@app.route("/api/select_video", methods=["POST"])
def api_select_video():
    """선택한 영상의 배경 제거 + Lottie 변환."""
    data = request.json or {}
    video_path = data.get("video_path", "")
    preset = data.get("preset", "original")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": f"영상 없음: {video_path}"}), 400

    backend, converter = _get_backend()
    stem = Path(video_path).stem
    base_stem = stem.replace("_attempt", "_att").split("_att")[0]

    # 1. 배경 제거
    transparent_dir = os.path.join(DIR_MOTION, f"{stem}_transparent")
    try:
        # fps는 vision_analyzer의 결과를 사용하는 것이 이상적이지만
        # 여기서는 기본값 16fps 사용
        fps = int(data.get("fps", 16))
        backend.bg_remover.remove_background(
            video_path=video_path,
            output_dir=transparent_dir,
            output_apng=True,
            output_webm=True,
            fps=fps,
        )
    except Exception as e:
        return jsonify({"error": f"배경 제거 실패: {e}"}), 500

    # 2. Lottie 변환
    lottie_path = os.path.join(transparent_dir, f"{stem}.json")
    try:
        converter.convert(
            png_dir=transparent_dir,
            output_path=lottie_path,
            fps=fps,
            preset=preset,
        )
    except Exception as e:
        return jsonify({"error": f"Lottie 변환 실패: {e}"}), 500

    info = converter.get_lottie_info(lottie_path)

    return jsonify({
        "transparent_dir": transparent_dir,
        "lottie_path": lottie_path,
        "lottie_info": info,
        "apng_path": os.path.join(transparent_dir, f"{stem}_transparent.apng"),
        "webm_path": os.path.join(transparent_dir, f"{stem}_transparent.webm"),
    })


# ── 생성 통계 API ────────────────────────────────────────────

@app.route("/api/stats/<stem>", methods=["GET"])
def api_stats(stem):
    """특정 이미지의 생성 통계 (validation_stats.txt 파싱)."""
    stats_path = os.path.join(OUTPUT_DIR, "validation_stats.txt")
    if not os.path.exists(stats_path):
        return jsonify({"error": "통계 파일 없음"}), 404

    records = []
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("image", "") == stem:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 결과/이슈 집계
    total = len(records)
    success_count = sum(1 for r in records if r.get("result") == "success")
    fail_count = total - success_count

    # 이슈 유형별 카운트
    issue_counts = {}
    for r in records:
        for issue in r.get("issues", []):
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    # attempt별 결과 (그래프용)
    attempt_results = []
    for i, r in enumerate(records):
        attempt_results.append({
            "attempt": i + 1,
            "result": r.get("result", "unknown"),
            "issues": r.get("issues", []),
            "vram_mb": r.get("vram_mb", 0),
            "remedy": r.get("remedy", ""),
        })

    return jsonify({
        "stem": stem,
        "total": total,
        "success": success_count,
        "fail": fail_count,
        "success_rate": round(success_count / total * 100, 1) if total > 0 else 0,
        "issue_counts": issue_counts,
        "attempts": attempt_results,
    })


# ── 전체 누적 통계 API ───────────────────────────────────────

@app.route("/api/stats_all", methods=["GET"])
def api_stats_all():
    """전체 누적 통계 요약."""
    stats_path = os.path.join(OUTPUT_DIR, "validation_stats.txt")
    if not os.path.exists(stats_path):
        return jsonify({"total": 0, "images": {}})

    images = {}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    rec = json.loads(line)
                    img = rec.get("image", "unknown")
                    if img not in images:
                        images[img] = {"total": 0, "success": 0, "fail": 0}
                    images[img]["total"] += 1
                    if rec.get("result") == "success":
                        images[img]["success"] += 1
                    else:
                        images[img]["fail"] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    total = sum(v["total"] for v in images.values())
    total_success = sum(v["success"] for v in images.values())

    return jsonify({
        "total": total,
        "success": total_success,
        "fail": total - total_success,
        "success_rate": round(total_success / total * 100, 1) if total > 0 else 0,
        "image_count": len(images),
        "images": images,
    })


# ── 파일 브라우저 API ─────────────────────────────────────────

@app.route("/api/browse", methods=["GET"])
def api_browse():
    """디렉토리 내용 반환 (파일 브라우저용)."""
    dir_path = request.args.get("dir", os.path.expanduser("~"))
    if not os.path.isdir(dir_path):
        return jsonify({"error": f"디렉토리 아님: {dir_path}"}), 400

    entries = []
    try:
        # 상위 디렉토리
        parent = os.path.dirname(dir_path)
        if parent != dir_path:
            entries.append({"name": "..", "path": parent, "type": "dir"})

        for name in sorted(os.listdir(dir_path)):
            full = os.path.join(dir_path, name)
            if name.startswith("."):
                continue
            if os.path.isdir(full):
                entries.append({"name": name + "/", "path": full, "type": "dir"})
            elif name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                size_kb = os.path.getsize(full) / 1024
                entries.append({
                    "name": name, "path": full, "type": "image",
                    "size_kb": round(size_kb, 1),
                })
    except PermissionError:
        return jsonify({"error": "접근 권한 없음"}), 403

    return jsonify({"dir": dir_path, "entries": entries})


# ── 키프레임 생성 API ────────────────────────────────────────

@app.route("/api/generate_keyframe", methods=["POST"])
def api_generate_keyframe():
    """suggested_action으로 키프레임 생성."""
    from image_pipeline.sprite_gen.wan_keyframe_generator import WanKeyframeGenerator
    data = request.json or {}
    action = data.get("suggested_action", "wobble")
    facing = data.get("facing_direction", "none")
    duration = data.get("duration_ms")

    gen = WanKeyframeGenerator()
    config = gen.generate(
        suggested_action=action,
        facing_direction=facing,
        duration_ms=int(duration) if duration else None,
    )
    return jsonify(config.to_dict())


# ── Lottie + Keyframe 통합 내보내기 ──────────────────────────

@app.route("/api/export_combined", methods=["POST"])
def api_export_combined():
    """Lottie에 키프레임을 베이크하여 통합 파일 다운로드."""
    from image_pipeline.sprite_gen.wan_lottie_baker import bake_keyframes

    data = request.json or {}
    lottie_path = data.get("lottie_path", "")
    keyframe_data = data.get("keyframe_data", {})
    output_name = data.get("output_name", "")

    if not os.path.isabs(lottie_path):
        anim_root = os.path.join(os.path.expanduser("~"), "anim_pipeline")
        lottie_path = os.path.join(anim_root, lottie_path)

    if not os.path.exists(lottie_path):
        return jsonify({"error": f"Lottie 파일 없음: {lottie_path}"}), 404
    if not keyframe_data or not keyframe_data.get("keyframes"):
        return jsonify({"error": "keyframe_data 없음"}), 400

    lottie_dir = os.path.dirname(lottie_path)
    stem = output_name or (Path(lottie_path).stem + "_combined")
    output_path = os.path.join(lottie_dir, f"{stem}.json")

    try:
        result_path = bake_keyframes(lottie_path, keyframe_data, output_path)
        return send_file(result_path, mimetype="application/json",
                         as_attachment=True, download_name=f"{stem}.json")
    except Exception as e:
        logger.error(f"[Export] 통합 실패: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export_lottie", methods=["POST"])
def api_export_lottie():
    """원본 Lottie JSON (모션만) 다운로드."""
    data = request.json or {}
    lottie_path = data.get("lottie_path", "")

    if not os.path.isabs(lottie_path):
        anim_root = os.path.join(os.path.expanduser("~"), "anim_pipeline")
        lottie_path = os.path.join(anim_root, lottie_path)

    if not os.path.exists(lottie_path):
        return jsonify({"error": f"Lottie 파일 없음: {lottie_path}"}), 404

    return send_file(lottie_path, mimetype="application/json",
                     as_attachment=True, download_name=Path(lottie_path).name)


# ── Stage 2: 모션 키프레임 판단 ──────────────────────────────

@app.route("/api/classify_motion", methods=["POST"])
def api_classify_motion():
    """Stage 2: 생성된 영상으로 키프레임 이동 필요 여부 판단."""
    data = request.json or {}
    video_path = data.get("video_path", "")
    image_path = data.get("image_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": f"영상 없음: {video_path}"}), 400

    backend, _ = _get_backend()
    result = backend.classify_post_motion(video_path, image_path)

    return jsonify({
        "needs_keyframe":     result.needs_keyframe,
        "travel_type":        result.travel_type.value,
        "travel_direction":   result.travel_direction.value,
        "confidence":         round(result.confidence, 3),
        "reason":             result.reason,
        "suggested_keyframe": result.suggested_keyframe,
    })


# ── 영상 첫 프레임 썸네일 ────────────────────────────────────

@app.route("/api/video_thumb/<path:filepath>", methods=["GET"])
def api_video_thumb(filepath):
    """영상의 첫 프레임을 PNG로 추출하여 반환."""
    import subprocess
    import tempfile

    if filepath.startswith("home/"):
        full_path = "/" + filepath
    elif filepath.startswith("outputs/"):
        anim_root = os.path.join(os.path.expanduser("~"), "anim_pipeline")
        full_path = os.path.join(anim_root, filepath)
    else:
        full_path = os.path.join(DIR_MOTION, filepath)

    if not os.path.exists(full_path):
        return jsonify({"error": "file not found"}), 404

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        subprocess.run([
            "ffmpeg", "-i", full_path, "-frames:v", "1",
            "-y", "-loglevel", "quiet", tmp_path,
        ], capture_output=True, timeout=10)
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            return send_file(tmp_path, mimetype="image/png")
        return jsonify({"error": "thumbnail extraction failed"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── 정적 파일 서빙 ───────────────────────────────────────────

@app.route("/api/files/<path:filepath>", methods=["GET"])
def api_files(filepath):
    """영상/이미지/JSON 파일 서빙."""
    # 절대 경로 복원
    if filepath.startswith("home/"):
        full_path = "/" + filepath
    elif os.path.isabs(filepath):
        full_path = filepath
    elif filepath.startswith("outputs/"):
        # 상대 경로 outputs/wan/... → anim_pipeline 루트 기준으로 해석
        anim_root = os.path.join(os.path.expanduser("~"), "anim_pipeline")
        full_path = os.path.join(anim_root, filepath)
    else:
        full_path = os.path.join(OUTPUT_DIR, filepath)

    if not os.path.exists(full_path):
        alt_path = os.path.join(OUTPUT_DIR, os.path.basename(filepath))
        if os.path.exists(alt_path):
            full_path = alt_path
        else:
            return jsonify({"error": f"file not found: {full_path}"}), 404

    ext = os.path.splitext(full_path)[1].lower()
    mime_map = {
        ".mp4": "video/mp4", ".webm": "video/webm",
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".json": "application/json", ".apng": "image/apng", ".gif": "image/gif",
    }
    return send_file(full_path, mimetype=mime_map.get(ext))


@app.route("/", methods=["GET"])
def index():
    """대시보드 HTML 서빙."""
    dashboard_path = os.path.join(
        os.path.dirname(__file__), "wan_dashboard.html"
    )
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    return "<h1>WAN Dashboard</h1><p>wan_dashboard.html not found</p>", 404


# ── 메인 ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)