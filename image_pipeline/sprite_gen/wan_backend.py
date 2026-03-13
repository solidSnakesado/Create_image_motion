"""
image_pipeline/sprite_gen/wan_backend.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    WAN I2V 자동화 파이프라인의 메인 오케스트레이터.

    전체 흐름:
        1. WanVisionAnalyzer  — LLM Vision으로 이미지 분석 → 액션 타입 결정
        2. 이미지 전처리       — 여백 추가 (액션별 headroom)
        3. ComfyUI API 호출   — WAN I2V 영상 생성 (randomize seed)
        4. WanValidator       — 결과 자동 검증
        5. 통과 시 저장 / 실패 시 재시도 (최대 max_retries)
        6. max_retries 초과 시 None 반환

ComfyUI API:
    ComfyUI는 /prompt 엔드포인트로 워크플로우 JSON을 받아 실행.
    /history/{prompt_id} 로 완료 여부 + 결과 파일명 조회.

환경 변수:
    COMFYUI_URL        : ComfyUI 서버 주소 (기본: http://127.0.0.1:8188)
    COMFYUI_ROOT       : ComfyUI 설치 루트 경로 (기본: ~/ComfyUI)
    WAN_WORKFLOW_PATH  : 워크플로우 JSON 전체 경로
                         기본: ~/ComfyUI/user/default/workflows/wan21_native_i2v_1.json
                         파일 존재 시 파일 로드 방식, 없으면 코드 빌드 방식(fallback)
    WAN_MODEL          : WAN GGUF 모델 파일명 (fallback 방식에서만 사용)
    CLIP_MODEL         : CLIP Vision 모델 파일명 (fallback 방식에서만 사용)
    VAE_MODEL          : VAE 모델 파일명 (fallback 방식에서만 사용)
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# wan_presets 불필요 — LLM이 모든 수치 직접 결정
from .wan_validator import WanValidator, ValidationResult
from .wan_vision_analyzer import WanVisionAnalyzer, VisionAnalysisResult
from .wan_ai_validator import WanAIValidator, AIValidationResult
from .wan_bg_remover import WanBgRemover
from .wan_mask_generator import WanMaskGenerator
from .wan_mode_classifier import WanModeClassifier, ProcessingMode, ModeClassification
from .wan_post_motion_classifier import WanPostMotionClassifier, PostMotionResult
from .wan_keyframe_generator import WanKeyframeGenerator, KeyframeAnimConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

MAX_RETRIES = 7
ATTEMPT_OFFSET = 0   # 서버에서 기존 영상 수만큼 오프셋 설정

COMFYUI_URL   = os.environ.get("COMFYUI_URL",   "http://127.0.0.1:8188")
WAN_MODEL     = os.environ.get("WAN_MODEL",     "wan2.1-i2v-14b-480p-Q3_K_S.gguf")
CLIP_MODEL    = os.environ.get("CLIP_MODEL",    "clip_vision_h.safetensors")
VAE_MODEL     = os.environ.get("VAE_MODEL",     "wan_2.1_vae.safetensors")

# ComfyUI 루트 디렉토리
# - 환경 변수 COMFYUI_ROOT 로 재정의 가능
# - 기본값: ~/ComfyUI (ComfyUI 표준 설치 경로)
COMFYUI_ROOT = os.environ.get(
    "COMFYUI_ROOT",
    os.path.expanduser("~/ComfyUI"),
)

# 워크플로우 파일 경로
# - 실제 위치: ComfyUI/user/default/workflows/wan21_native_i2v_1.json
# - 환경 변수 WAN_WORKFLOW_PATH 로 전체 경로 재정의 가능
# - 파일이 존재하면 파일 로드 + 파라미터 주입 방식으로 자동 전환
# - 파일 없으면 코드 빌드 방식(fallback) 사용
WAN_WORKFLOW_PATH = os.environ.get(
    "WAN_WORKFLOW_PATH",
    os.path.join(
        COMFYUI_ROOT,
        "user", "default", "workflows",   # ComfyUI 워크플로우 저장 경로
        "wan21_native_i2v_1.json",        # 만족스러운 결과를 낸 워크플로우
    )
)


# ─────────────────────────────────────────────────────────────
# 결과 모델
# ─────────────────────────────────────────────────────────────

class WanResult:
    """WAN 생성 최종 결과."""
    def __init__(
        self,
        success:         bool,
        video_path:      str | None     = None,
        analysis:        VisionAnalysisResult | None = None,
        validation:      ValidationResult | None     = None,
        attempts:        int  = 0,
        seed:            int  = 0,
        mode:            ModeClassification | None = None,
        post_motion:     PostMotionResult | None   = None,
        keyframe_config: KeyframeAnimConfig | None = None,
    ):
        self.success         = success
        self.video_path      = video_path
        self.analysis        = analysis
        self.validation      = validation
        self.attempts        = attempts
        self.seed            = seed
        self.mode            = mode
        self.post_motion     = post_motion
        self.keyframe_config = keyframe_config

    def __str__(self) -> str:
        if self.mode and self.mode.processing_mode == ProcessingMode.KEYFRAME_ONLY:
            kf_info = ""
            if self.keyframe_config:
                kf_info = f", kf={self.keyframe_config.animation_type}/{self.keyframe_config.duration_ms}ms"
            return (
                f"✅ WanResult(mode=KEYFRAME_ONLY, "
                f"facing={self.mode.facing_direction.value}, "
                f"action={self.mode.suggested_action}{kf_info})"
            )
        if self.success:
            pm = ""
            if self.post_motion and self.post_motion.needs_keyframe:
                pm = f", travel={self.post_motion.travel_direction.value}"
            return (
                f"✅ WanResult(video={self.video_path}, "
                f"action={self.analysis.action_desc}, "
                f"attempts={self.attempts}{pm})"
            )
        return f"❌ WanResult(failed after {self.attempts} attempts)"


# ─────────────────────────────────────────────────────────────
# 이미지 전처리
# ─────────────────────────────────────────────────────────────

def _white_anchor(image: Image.Image, tolerance: int = 30) -> Image.Image:
    """
    White Anchor 전처리.

    역할:
        1. 배경 클리닝: 테두리 기준 배경색과 유사한 픽셀(tolerance 이내)을
           완벽한 순백색 RGB(255,255,255)으로 강제 교체.
           → WAN VAE 인코딩 시 배경이 일관된 latent 값으로 매핑됨.
        2. 캐릭터 가장자리 선명화 (UnsharpMask):
           캐릭터-배경 경계를 더 뚜렷하게 → WAN이 배경을 고정 캔버스로 인식.

    적용 조건:
        배경이 밝은 색(평균 > 200)일 때만 적용.
        어두운 배경 이미지에는 불필요하며 캐릭터를 손상시킬 수 있음.
    """
    import numpy as np
    from PIL import ImageFilter
    from scipy import ndimage

    arr = np.array(image.convert("RGB"))
    H, W = arr.shape[:2]

    # 테두리 평균색 계산
    bs = max(3, H // 20)
    border = np.concatenate([
        arr[:bs, :].reshape(-1, 3),
        arr[-bs:, :].reshape(-1, 3),
        arr[:, :bs].reshape(-1, 3),
        arr[:, -bs:].reshape(-1, 3),
    ])
    bg_mean = border.mean(axis=0)

    # 밝은 배경일 때만 적용 (평균 > 200)
    if bg_mean.mean() < 200:
        return image

    # 배경 마스크: 테두리와 연결된 배경색 픽셀
    is_bg_color = np.all(np.abs(arr.astype(int) - bg_mean) < tolerance, axis=2)
    labeled, _ = ndimage.label(is_bg_color)
    border_labels = (
        set(labeled[0, :].tolist()) | set(labeled[-1, :].tolist()) |
        set(labeled[:, 0].tolist()) | set(labeled[:, -1].tolist())
    )
    border_labels.discard(0)
    bg_mask = np.zeros((H, W), dtype=bool)
    for lbl in border_labels:
        bg_mask |= (labeled == lbl)

    # 배경 픽셀 → 순백색(255)
    result = arr.copy()
    result[bg_mask] = 255

    # 캐릭터 가장자리 선명화 (UnsharpMask: radius=1, percent=120, threshold=3)
    sharpened = Image.fromarray(result).filter(
        ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)
    )
    # 배경 영역은 선명화 후에도 255 유지 (선명화가 테두리 픽셀을 회색으로 만들 수 있음)
    s_arr = np.array(sharpened)
    s_arr[bg_mask] = 255

    return Image.fromarray(s_arr)


def preprocess_image(
    image_path: str,
    preset:     WanPreset,
    output_path: str,
) -> str:
    """
    원본 이미지에 액션별 여백을 추가하여 저장.

    개구리(jump): 상단 여백 크게 — 점프 공간 확보
    새(fly)     : 상하 여백 균등 — 날개 공간 확보
    물고기(swim): 상하좌우 균등 — 꼬리 공간 확보

    Returns:
        전처리된 이미지 저장 경로
    """
    src = Image.open(image_path).convert("RGBA")
    ow, oh = src.size

    W, H = width, height

    # 오브젝트 리사이즈
    target_w = int(W * preset.scale)
    ratio    = target_w / ow
    target_h = int(oh * ratio)
    src_resized = src.resize((target_w, target_h), Image.LANCZOS)

    # 캔버스 생성 (흰색)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    # 배치: 상단 여백 확보, 수평 중앙
    x = (W - target_w) // 2
    y = int(H * preset.headroom_top)

    # 하단 여백 체크: 오브젝트가 캔버스를 넘으면 y 조정
    if y + target_h > H - int(H * preset.headroom_bottom):
        y = max(0, H - target_h - int(H * preset.headroom_bottom))

    canvas.paste(src_resized, (x, y), src_resized)
    canvas_rgb = canvas.convert("RGB")
    canvas_rgb = _white_anchor(canvas_rgb)
    canvas_rgb.save(output_path)

    logger.info(
        f"[Preprocess] {ow}×{oh} → {target_w}×{target_h} "
        f"배치=({x},{y}) 캔버스={W}×{H}"
    )
    return output_path


def preprocess_image_simple(
    image_path:  str,
    output_path: str,
    width:       int   = 480,
    height:      int   = 480,
    scale:       float = 0.65,
    headroom_top:    float = 0.18,
    headroom_bottom: float = 0.15,
) -> str:
    """
    프리셋 없이 고정 수치로 이미지 전처리.
    LLM이 프리셋을 사용하지 않는 새 파이프라인용.

    원본이 캔버스(480×480) 이내이면 축소 없이 원본 크기 유지.
    원본이 캔버스를 초과하면 scale 비율로 축소.
    """
    src = Image.open(image_path).convert("RGBA")
    ow, oh = src.size

    if ow <= width and oh <= height:
        # 원본이 캔버스 이내 → 축소 없이 원본 크기 유지
        target_w, target_h = ow, oh
        src_resized = src
    else:
        # 원본이 캔버스 초과 → scale 적용하여 축소
        target_w = int(width * scale)
        ratio    = target_w / ow
        target_h = int(oh * ratio)
        # 축소 후에도 캔버스를 초과하면 캔버스에 맞춤
        if target_h > height:
            target_h = int(height * scale)
            ratio    = target_h / oh
            target_w = int(ow * ratio)
        src_resized = src.resize((target_w, target_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    x = (width - target_w) // 2
    y = int(height * headroom_top)
    if y + target_h > height - int(height * headroom_bottom):
        y = max(0, height - target_h - int(height * headroom_bottom))

    canvas.paste(src_resized, (x, y), src_resized)
    result = _white_anchor(canvas.convert("RGB"))
    result.save(output_path)

    scaled_str = "원본유지" if (target_w == ow and target_h == oh) else "축소"
    logger.info(
        f"[Preprocess] simple {ow}×{oh} → {target_w}×{target_h} ({scaled_str}) "
        f"배치=({x},{y}) 캔버스={width}×{height}"
    )
    return output_path


# ─────────────────────────────────────────────────────────────
# 검증 실패 통계 추적기
# ─────────────────────────────────────────────────────────────

class _ValidationStats:
    """
    검증 실패 항목 + 보완 조치를 이미지별·누적으로 추적하고 파일에 기록.

    파일 기록 형식 (validation_stats.txt):
        [2026-03-10 12:34:56] IMAGE: frog_01
          attempt 1: metric_fail | ghosting  [AI: unnatural_movement] → ai_adjust (...) [VRAM:8133MB]
          attempt 2: metric_fail | too_slow  → seed_retry (...) [VRAM:8145MB]
          attempt 3: success     |           [VRAM:8150MB]
          ---

    터미널 출력 (이미지별 + 누적 통계):
        ──── [통계] frog_01 — 총 3회 시도 ────
          수치실패: 2 (66.7%)  ...
        ════ [누적 통계] 이미지 2장 | 총 10회 ════
          ...

    load_history(): 기존 파일 파싱 (구 포맷·신 포맷 모두 호환)
    """

    # attempt 라인 파싱용 정규식
    _ATTEMPT_RE = re.compile(
        r"attempt\s+(\d+):\s*(\S+)\s*\|\s*([^[\]→]*?)"
        r"(?:\[AI:\s*([^\]]*)\])?"
        r"(?:\s*→\s*(\S+)\s*(?:\(([^)]*)\))?)?"
        r"(?:\s*\[VRAM:(\d+)MB\])?"
        r"\s*$"
    )

    def __init__(self, output_dir: str) -> None:
        self._file_path = os.path.join(output_dir, "validation_stats.txt")
        # 현재 이미지용 기록
        self._current_image: str = ""
        self._current_records: list[dict] = []
        # 전체 누적 통계
        self._total_attempts: int = 0
        self._total_metric_fails: int = 0
        self._total_ai_fails: int = 0
        self._total_success: int = 0
        self._total_ai_issue_attempts: int = 0  # AI이슈가 발생한 시도 수
        self._total_metric_items: Counter = Counter()
        self._total_ai_items: Counter = Counter()
        self._total_remedies: Counter = Counter()
        self._total_images: int = 0

        # 기존 파일에서 누적 통계 복원
        self._restore_cumulative_from_file()

    def load_history(self, image_name: str | None = None) -> dict:
        """
        기존 validation_stats.txt를 파싱하여 이력 반환.
        구 포맷(요약 통계 포함)과 신 포맷(attempt + 구분선만) 모두 호환.

        Args:
            image_name: 특정 이미지만 조회 (None이면 전체)

        Returns:
            {
                "images": {
                    "frog_01": {
                        "attempts": [
                            {
                                "type": "metric_fail",
                                "issues": ["ghosting"],
                                "ai_issues": ["unnatural_movement"],
                                "remedy": "ai_adjust",
                                "remedy_detail": "fps:14→16, ...",
                                "vram_mb": 8133,
                            }, ...
                        ],
                    }, ...
                },
                "total_attempts": 28,
                "total_images": 4,
            }
        """
        result: dict = {"images": {}, "total_attempts": 0, "total_images": 0}

        if not os.path.exists(self._file_path):
            return result

        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return result

        current_img = None
        current_attempts: list[dict] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 이미지 헤더: [2026-03-10 12:19:56] IMAGE: frog_01
            if stripped.startswith("[") and "IMAGE:" in stripped:
                # 이전 이미지 저장
                if current_img and current_attempts:
                    result["images"][current_img] = {
                        "attempts": current_attempts,
                    }
                current_img = stripped.split("IMAGE:")[-1].strip()
                current_attempts = []
                continue

            # attempt 라인 파싱
            m = self._ATTEMPT_RE.search(stripped)
            if m:
                issues_raw = m.group(3).strip().rstrip(",")
                issues = [x.strip() for x in issues_raw.split(",") if x.strip()]
                ai_raw = m.group(4)
                ai_issues = (
                    [x.strip() for x in ai_raw.split(",") if x.strip()]
                    if ai_raw else []
                )
                vram = int(m.group(7)) if m.group(7) else None

                current_attempts.append({
                    "type": m.group(2),
                    "issues": issues,
                    "ai_issues": ai_issues,
                    "remedy": m.group(5) or None,
                    "remedy_detail": m.group(6) or None,
                    "vram_mb": vram,
                })
                continue

            # 구 포맷의 요약 통계·구분선은 무시

        # 마지막 이미지 저장
        if current_img and current_attempts:
            result["images"][current_img] = {
                "attempts": current_attempts,
            }

        result["total_images"] = len(result["images"])
        result["total_attempts"] = sum(
            len(img["attempts"]) for img in result["images"].values()
        )

        # 특정 이미지만 필터링
        if image_name and image_name in result["images"]:
            return {
                "images": {image_name: result["images"][image_name]},
                "total_attempts": len(result["images"][image_name]["attempts"]),
                "total_images": 1,
            }

        return result

    def _restore_cumulative_from_file(self) -> None:
        """기존 validation_stats.txt에서 누적 통계를 복원."""
        history = self.load_history()
        if not history["images"]:
            return

        for img_name, img_data in history["images"].items():
            self._total_images += 1
            for a in img_data["attempts"]:
                self._total_attempts += 1
                rtype = a["type"]
                issues = a.get("issues", [])
                ai_issues = a.get("ai_issues", [])
                remedy = a.get("remedy")

                if rtype == "metric_fail":
                    self._total_metric_fails += 1
                    self._total_metric_items.update(issues)
                    if ai_issues:
                        self._total_ai_items.update(ai_issues)
                        self._total_ai_issue_attempts += 1
                elif rtype == "ai_fail":
                    self._total_ai_fails += 1
                    self._total_ai_items.update(issues)
                    self._total_ai_issue_attempts += 1
                elif rtype == "success":
                    self._total_success += 1

                if remedy:
                    self._total_remedies[remedy] += 1

        logger.info(
            f"  [통계] 기존 이력 복원: {self._total_images}장 "
            f"{self._total_attempts}회 (파일: {self._file_path})"
        )

    def record(
        self,
        image_name:  str,
        result_type: str,
        issues:      list[str],
        ai_issues:   list[str] | None = None,
        remedy:      str | None = None,
        remedy_detail: str | None = None,
        vram_mb:     int | None = None,
    ) -> None:
        """
        한 번의 시도 결과를 기록.

        Args:
            image_name:    이미지 식별자 (stem)
            result_type:   "metric_fail" | "ai_fail" | "success"
            issues:        실패 항목 리스트 (성공 시 빈 리스트)
            ai_issues:     수치 실패 시 함께 발생한 AI 검증 이슈 (별도 추적)
            remedy:        보완 조치 종류
                           "seed_retry" | "ai_adjust" | "action_switch" | "soft_pass" | None
            remedy_detail: 보완 조치 상세 내용 (예: "fps:14→16, negative:身体晃动")
            vram_mb:       시도 시작 시점 VRAM 사용량 (MB)
        """
        if self._current_image != image_name:
            self._flush_pending_attempt()
            self._current_image = image_name
            self._current_records = []

        self._flush_pending_attempt()

        self._current_records.append({
            "type": result_type,
            "issues": issues,
            "ai_issues": ai_issues or [],
            "remedy": remedy,
            "remedy_detail": remedy_detail,
            "vram_mb": vram_mb,
        })

        # 누적 카운터 업데이트
        self._total_attempts += 1
        if result_type == "metric_fail":
            self._total_metric_fails += 1
            self._total_metric_items.update(issues)
            if ai_issues:
                self._total_ai_items.update(ai_issues)
                self._total_ai_issue_attempts += 1
        elif result_type == "ai_fail":
            self._total_ai_fails += 1
            self._total_ai_items.update(issues)
            self._total_ai_issue_attempts += 1
        elif result_type == "success":
            self._total_success += 1

        if remedy:
            self._total_remedies[remedy] += 1

        # 현재 시도는 pending 상태 (remedy/ai_issues가 나중에 업데이트될 수 있음)
        self._pending_flush = True

    def _flush_pending_attempt(self) -> None:
        """
        대기 중인 마지막 시도를 파일에 기록.
        remedy/ai_issues가 모두 확정된 후에 호출됨.
        """
        if not getattr(self, '_pending_flush', False):
            return
        if not self._current_records:
            return

        self._pending_flush = False
        r = self._current_records[-1]
        attempt_num = len(self._current_records)

        # 새 이미지 첫 시도면 헤더 추가
        header = ""
        if attempt_num == 1:
            header = (
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"IMAGE: {self._current_image}\n"
            )

        issues_str = ", ".join(r["issues"]) if r["issues"] else ""
        ai_str = ""
        if r.get("ai_issues"):
            ai_str = f" [AI: {', '.join(r['ai_issues'])}]"
        remedy_str = ""
        if r.get("remedy"):
            remedy_str = f" → {r['remedy']}"
            if r.get("remedy_detail"):
                remedy_str += f" ({r['remedy_detail']})"
        vram_str = ""
        if r.get("vram_mb") is not None:
            vram_str = f" [VRAM:{r['vram_mb']}MB]"

        line = f"  attempt {attempt_num}: {r['type']:<12} | {issues_str:<40}{ai_str}{remedy_str}{vram_str}\n"

        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(header + line)
        except Exception:
            pass

    def _update_last_remedy(self, remedy: str, detail: str = "") -> None:
        """마지막 기록의 보완 조치를 업데이트 (파일 저장은 flush 시)."""
        if self._current_records:
            self._current_records[-1]["remedy"] = remedy
            self._current_records[-1]["remedy_detail"] = detail
            self._total_remedies[remedy] += 1

    def _update_last_ai_issues(self, ai_issues: list[str]) -> None:
        """마지막 기록(수치 실패)에 AI 검증 이슈를 추가 (파일 저장은 flush 시)."""
        if self._current_records and ai_issues:
            # 이전에 ai_issues가 없었으면 카운터 증가
            if not self._current_records[-1].get("ai_issues"):
                self._total_ai_issue_attempts += 1
            self._current_records[-1]["ai_issues"] = ai_issues
            self._total_ai_items.update(ai_issues)

    def print_image_summary(self, image_name: str) -> None:
        """현재 이미지의 검증 실패 비율을 로그로 출력."""
        records = self._current_records
        if not records:
            return

        self._total_images += 1
        total = len(records)
        metric_fails = sum(1 for r in records if r["type"] == "metric_fail")
        ai_fails = sum(1 for r in records if r["type"] == "ai_fail")
        successes = sum(1 for r in records if r["type"] == "success")

        metric_items: Counter = Counter()
        ai_items: Counter = Counter()
        remedy_counts: Counter = Counter()
        for r in records:
            if r["type"] == "metric_fail":
                metric_items.update(r["issues"])
                # 수치 실패 시 함께 발생한 AI 이슈도 수집
                if r.get("ai_issues"):
                    ai_items.update(r["ai_issues"])
            elif r["type"] == "ai_fail":
                ai_items.update(r["issues"])
            if r["remedy"]:
                remedy_counts[r["remedy"]] += 1

        pct = lambda n, d=total: f"{n/d*100:.1f}%" if d > 0 else "0%"

        # AI 이슈가 발생한 시도 수 (수치실패+AI이슈 or 순수 AI실패)
        ai_issue_attempts = sum(
            1 for r in records
            if r["type"] == "ai_fail" or r.get("ai_issues")
        )

        lines = [
            f"\n{'─'*60}",
            f"  [통계] {image_name} — 총 {total}회 시도",
            f"    수치실패: {metric_fails} ({pct(metric_fails)})",
        ]
        for item, cnt in metric_items.most_common():
            lines.append(f"      - {item}: {cnt} ({pct(cnt)})")

        lines.append(f"    AI검증 단독 실패: {ai_fails} ({pct(ai_fails)})")
        lines.append(f"    AI이슈 발생: {ai_issue_attempts}회 ({pct(ai_issue_attempts)})")
        for item, cnt in ai_items.most_common():
            lines.append(f"      - {item}: {cnt} ({pct(cnt)})")

        lines.append(f"    성공: {successes} ({pct(successes)})")

        if remedy_counts:
            remedy_total = sum(remedy_counts.values())
            lines.append(f"    보완 조치: (총 {remedy_total}회)")
            for rem, cnt in remedy_counts.most_common():
                lines.append(f"      - {rem}: {cnt} ({pct(cnt, remedy_total)})")

        lines.append(f"{'─'*60}")
        logger.info("\n".join(lines))

    def print_cumulative_summary(self) -> None:
        """전체 누적 통계를 로그로 출력."""
        total = self._total_attempts
        if total == 0:
            return

        pct = lambda n, d=total: f"{n/d*100:.1f}%" if d > 0 else "0%"

        lines = [
            f"\n{'═'*60}",
            f"  [누적 통계] 이미지 {self._total_images}장 | 총 {total}회 시도",
            f"    수치실패: {self._total_metric_fails} ({pct(self._total_metric_fails)})",
        ]
        for item, cnt in self._total_metric_items.most_common():
            lines.append(f"      - {item}: {cnt} ({pct(cnt)})")

        lines.append(
            f"    AI검증 단독 실패: {self._total_ai_fails} ({pct(self._total_ai_fails)})"
        )
        lines.append(
            f"    AI이슈 발생: {self._total_ai_issue_attempts}회 "
            f"({pct(self._total_ai_issue_attempts)})"
        )
        for item, cnt in self._total_ai_items.most_common():
            lines.append(f"      - {item}: {cnt} ({pct(cnt)})")

        lines.append(f"    성공: {self._total_success} ({pct(self._total_success)})")

        if self._total_remedies:
            remedy_total = sum(self._total_remedies.values())
            lines.append(f"    보완 조치: (총 {remedy_total}회)")
            for rem, cnt in self._total_remedies.most_common():
                pct_r = lambda n: f"{n/remedy_total*100:.1f}%" if remedy_total > 0 else "0%"
                lines.append(f"      - {rem}: {cnt} ({pct_r(cnt)})")

        lines.append(f"{'═'*60}")
        logger.info("\n".join(lines))

    def save_to_file(self) -> None:
        """마지막 pending 시도를 flush하고 구분선 추가.
        요약 통계는 터미널(print_image_summary/print_cumulative_summary)에만 출력.
        """
        # 마지막 시도 flush (루프 종료 시 pending 상태일 수 있음)
        self._flush_pending_attempt()

        if not self._current_records:
            return

        # 파일에 구분선만 추가
        try:
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write("  ---\n")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# ComfyUI API 클라이언트
# ─────────────────────────────────────────────────────────────

class ComfyUIClient:
    """ComfyUI HTTP API 클라이언트."""

    def __init__(self, base_url: str = COMFYUI_URL) -> None:
        self.base_url  = base_url.rstrip("/")
        self.client_id = f"wan_backend_{random.randint(0, 99999):05d}"

    def upload_image(self, image_path: str) -> str:
        """
        이미지를 ComfyUI input 폴더에 업로드.
        Returns: 업로드된 파일명
        """
        import urllib.request
        import mimetypes

        filename = Path(image_path).name
        with open(image_path, "rb") as f:
            data = f.read()

        boundary = "----WanBackendBoundary"
        # overwrite=true: 동일 파일명 존재 시 덮어쓰기 (캐시 방지)
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        uploaded_name = result.get("name", filename)
        logger.info(f"[ComfyUI] 이미지 업로드 완료: {uploaded_name}")
        return uploaded_name

    def queue_prompt(self, workflow: dict) -> str:
        """
        워크플로우를 큐에 추가.
        Returns: prompt_id
        """
        payload = json.dumps({
            "prompt":    workflow,
            "client_id": self.client_id,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())

        prompt_id = result["prompt_id"]
        logger.info(f"[ComfyUI] 큐 추가 완료: prompt_id={prompt_id}")
        return prompt_id


    def wait_for_completion(
        self,
        prompt_id: str,
        timeout:   int = 1800,
        interval:  float = 2.0,
    ) -> dict:
        """
        생성 완료까지 폴링 대기. 프로그레스 바 출력.
        Returns: history 딕셔너리
        """
        import sys
        start      = time.time()
        bar_width  = 30
        last_step  = 0
        last_total = 0

        print("", flush=True)

        while True:
            elapsed = time.time() - start
            if elapsed > timeout:
                print()
                raise TimeoutError(f"ComfyUI 타임아웃 ({timeout}s)")

            # ── 완료 확인 ────────────────────────────────────
            url = f"{self.base_url}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read())

            if prompt_id in history:
                entry = history[prompt_id]
                logger.debug(f"[ComfyUI] history status: {entry.get('status', {})}")
                # status 확인 - 완료/오류 모두 처리
                status = entry.get("status", {})
                status_str = status.get("status_str", "")
                completed  = status.get("completed", False)

                if completed or status_str in ("success", "error", ""):
                    filled = "█" * bar_width
                    print(
                        f"\r  [{filled}] 100% "
                        f"완료 ({elapsed:.1f}s)          ",
                        flush=True,
                    )
                    print()
                    logger.info(f"[ComfyUI] 생성 완료 ({elapsed:.1f}s)")
                    return entry
                # history에 있지만 아직 실행 중인 경우 계속 대기

            # ── 진행률 조회 ───────────────────────────────────
            try:
                prog_url = f"{self.base_url}/queue"
                with urllib.request.urlopen(prog_url, timeout=5) as resp:
                    queue_data = json.loads(resp.read())

                # 현재 실행 중인 항목에서 진행률 추출
                running = queue_data.get("queue_running", [])
                step, total = last_step, last_total

                for item in running:
                    # item 구조: [number, prompt_id, prompt, extra, ...]
                    if len(item) > 1 and item[1] == prompt_id:
                        # /progress 엔드포인트로 상세 진행률 조회
                        try:
                            p_url = f"{self.base_url}/progress"
                            with urllib.request.urlopen(p_url, timeout=5) as pr:
                                prog = json.loads(pr.read())
                                step  = prog.get("value", 0)
                                total = prog.get("max", 0) or last_total
                                last_step  = step
                                last_total = total
                        except Exception:
                            pass
                        break

                # 프로그레스 바 출력
                if total > 0:
                    pct    = step / total
                    filled = int(bar_width * pct)
                    bar    = "█" * filled + "░" * (bar_width - filled)
                    print(
                        f"\r  [{bar}] {int(pct*100):3d}% "
                        f"({step}/{total} steps) "
                        f"{elapsed:.0f}s 경과",
                        end="", flush=True,
                    )
                else:
                    # 진행률 미확인 - 스피너
                    spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
                    spin    = spinner[int(elapsed) % len(spinner)]
                    dots    = "." * (int(elapsed) % 4)
                    print(
                        f"\r  {spin} 생성 중{dots:<4} {elapsed:.0f}s 경과",
                        end="", flush=True,
                    )

            except Exception:
                # 큐 조회 실패 시 스피너만
                spinner = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
                spin    = spinner[int(elapsed) % len(spinner)]
                print(
                    f"\r  {spin} 생성 중... {elapsed:.0f}s 경과",
                    end="", flush=True,
                )

            time.sleep(interval)

    def download_video(self, history: dict, output_path: str) -> str:
        """
        history에서 비디오 파일을 다운로드하여 저장.
        Returns: 저장된 파일 경로
        """
        # 출력 노드에서 비디오 파일명 추출
        video_filename = None
        for node_output in history.get("outputs", {}).values():
            if "gifs" in node_output:
                for item in node_output["gifs"]:
                    if item.get("filename", "").endswith(".mp4"):
                        video_filename = item["filename"]
                        subfolder = item.get("subfolder", "")
                        break
            if video_filename:
                break

        if not video_filename:
            raise ValueError("history에서 비디오 파일을 찾을 수 없습니다")

        # 다운로드
        params = urllib.parse.urlencode({
            "filename":  video_filename,
            "subfolder": subfolder if subfolder else "",
            "type":      "output",
        })
        url = f"{self.base_url}/view?{params}"

        with urllib.request.urlopen(url, timeout=60) as resp:
            video_data = resp.read()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(video_data)

        logger.info(f"[ComfyUI] 비디오 다운로드 완료: {output_path}")
        return output_path

    def free_memory(self) -> None:
        """
        ComfyUI VRAM + 시스템 RAM 초기화.
        1. ComfyUI /free API → 모델 언로드 + CUDA 캐시 해제
        2. Python gc.collect() → Python 레벨 메모리 해제
        3. torch.cuda.empty_cache() → CUDA 메모리 풀 정리

        12GB VRAM + 32GB RAM 환경에서 WAN GGUF 연속 생성 시
        VRAM은 안정적이지만 RAM이 누적되는 문제(ComfyUI #11775)를 완화.
        """
        try:
            # ── 1. ComfyUI /free API 호출 ──────────────────────
            vram_before = self._get_vram_usage()
            ram_before = self._get_ram_usage()

            data = json.dumps({
                "unload_models": True,
                "free_memory": True,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/free",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)

            # ── 2. Python GC + torch 캐시 정리 ────────────────
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except ImportError:
                pass

            time.sleep(1)

            # ── 3. 결과 로그 ──────────────────────────────────
            vram_after = self._get_vram_usage()
            ram_after = self._get_ram_usage()

            parts = []
            if vram_before and vram_after:
                vram_freed = vram_before["used"] - vram_after["used"]
                parts.append(
                    f"VRAM: {vram_before['used']}→{vram_after['used']}MB "
                    f"(해제:{vram_freed}MB)"
                )
            if ram_before and ram_after:
                ram_freed = ram_before["used_gb"] - ram_after["used_gb"]
                parts.append(
                    f"RAM: {ram_before['used_gb']:.1f}→{ram_after['used_gb']:.1f}GB "
                    f"(해제:{ram_freed:.1f}GB) [{ram_after['percent']}%]"
                )
                # RAM 사용률 경고
                if ram_after["percent"] > 85:
                    logger.warning(
                        f"  ⚠ RAM 사용률 {ram_after['percent']}% — "
                        f"ComfyUI 재시작 권장 (pkill -f ComfyUI/main.py)"
                    )

            if parts:
                logger.info(f"[ComfyUI] 메모리 초기화 완료 — {' | '.join(parts)}")
            else:
                logger.info("[ComfyUI] 메모리 초기화 완료")

        except Exception as e:
            logger.warning(f"[ComfyUI] 메모리 초기화 실패 (생성에는 영향 없음): {e}")

    def _get_ram_usage(self) -> dict | None:
        """시스템 RAM 사용량 조회."""
        try:
            import subprocess
            result = subprocess.run(
                ["free", "-b"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # "Mem:" 라인 파싱
                for line in lines:
                    if line.startswith("Mem:"):
                        parts = line.split()
                        total = int(parts[1])
                        used = int(parts[2])
                        return {
                            "total_gb": total / (1024**3),
                            "used_gb": used / (1024**3),
                            "percent": int(used / total * 100),
                        }
        except Exception:
            pass
        return None

    def _get_vram_usage(self) -> dict | None:
        """nvidia-smi로 현재 VRAM 사용량 조회."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                used, total = result.stdout.strip().split(", ")
                return {"used": int(used), "total": int(total)}
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────
# 워크플로우 빌더
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# 워크플로우 파라미터 주입 매핑
# ─────────────────────────────────────────────────────────────
# wan21_native_i2v_1.json 의 각 노드 class_type → 주입할 파라미터 매핑.
# 워크플로우 파일이 바뀌어도 이 매핑만 업데이트하면 됨.
#
# 형식: { "class_type": { "input_key": value, ... }, ... }
# 같은 class_type 이 여러 노드에 있을 경우 모든 노드에 적용됨.
# (CLIPTextEncode positive/negative 는 node_id 로 구분 — 아래 참고)
#
# ★ POSITIVE / NEGATIVE 주입 방식:
#   - CLIPTextEncode 노드가 2개 (positive / negative)
#   - "positive_node_id" 와 "negative_node_id" 로 node_id 직접 지정
#   - 지정하지 않으면 첫 번째 CLIPTextEncode = positive, 두 번째 = negative

WORKFLOW_INJECT_MAP = {
    # 시드 — KSampler 노드
    "KSampler": {
        "seed": None,          # 런타임에 주입
    },
    # 프레임 레이트 — VHS_VideoCombine 노드
    "VHS_VideoCombine": {
        "frame_rate": None,    # 런타임에 주입
    },
    # 입력 이미지 — LoadImage 노드
    "LoadImage": {
        "image": None,         # 런타임에 주입
    },
}

# CLIPTextEncode positive/negative 노드 ID
# wan21_native_i2v_1.json 기준 (ComfyUI에서 직접 확인한 값)
POSITIVE_CLIP_NODE_ID = None  # None 이면 자동 감지 (첫 번째 CLIPTextEncode)
NEGATIVE_CLIP_NODE_ID = None  # None 이면 자동 감지 (두 번째 CLIPTextEncode)


def _gui_workflow_to_api(raw: dict) -> dict:
    """
    ComfyUI GUI 저장 포맷 -> API 포맷 변환.

    GUI format : {"nodes": [{id, type, widgets_values, inputs, ...}], "links": [...]}
    API format : {"1": {"class_type": "...", "inputs": {"key": value_or_link}}, ...}

    KSampler widgets_values 순서:
        [seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]
    VHS_VideoCombine widgets_values: dict 형태
    """
    WIDGET_KEYS: dict = {
        "CLIPLoader":       ["clip_name", "type", "device"],
        "CLIPVisionLoader": ["clip_name"],
        "VAELoader":        ["vae_name"],
        "UnetLoaderGGUF":   ["unet_name"],
        "CLIPTextEncode":   ["text"],
        "CLIPVisionEncode": ["crop"],
        "WanImageToVideo":  ["width", "height", "length", "batch_size"],
        "KSampler":         ["seed", "control_after_generate", "steps", "cfg",
                             "sampler_name", "scheduler", "denoise"],
        "LoadImage":        ["image", "upload"],
        "VAEDecode":        [],
        "VHS_VideoCombine": [],
    }

    # links: link_id -> [src_node_id_str, src_slot]
    link_by_id: dict = {}
    for lk in raw.get("links", []):
        link_by_id[lk[0]] = [str(lk[1]), lk[2]]

    api_workflow: dict = {}
    for node in raw["nodes"]:
        node_id = str(node["id"])
        ntype   = node["type"]
        wv      = node.get("widgets_values", [])
        ninputs = node.get("inputs", [])

        inputs: dict = {}

        # widgets_values -> inputs 키 매핑
        if ntype == "VHS_VideoCombine" and isinstance(wv, dict):
            for k, v in wv.items():
                if k != "videopreview":
                    inputs[k] = v
            # save_output 강제 true (빈 문자열 방지)
            inputs["save_output"] = True
            # loop_count=0: 루프 제거 → 루프 경계 열화 방지
            inputs["loop_count"] = 0
        else:
            for i, key in enumerate(WIDGET_KEYS.get(ntype, [])):
                if i < len(wv):
                    inputs[key] = wv[i]

        # 링크 연결 복원
        for inp in ninputs:
            link_id = inp.get("link")
            if link_id is None:
                continue
            src = link_by_id.get(link_id)
            if src:
                inputs[inp["name"]] = src

        api_workflow[node_id] = {"class_type": ntype, "inputs": inputs}

    return api_workflow


def load_workflow_from_file(
    workflow_path: str,
    image_filename: str,
    positive:       str,
    negative:       str,
    frame_rate:     int,
    seed:           int,
    width:          int = 480,
    height:         int = 480,
    steps:          int = 20,
    pingpong:       bool = False,
) -> dict:
    """
    워크플로우 JSON 파일 로드 후 동적 파라미터 주입.
    GUI format / API format 모두 자동 지원.
    주입: seed, frame_rate, image, positive text, negative text
    나머지(steps, cfg, sampler 등)는 파일 저장값 유지.
    """
    with open(workflow_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "nodes" in raw:
        logger.info("[Workflow] GUI format 감지 -> API format 변환")
        workflow = _gui_workflow_to_api(raw)
    else:
        logger.info("[Workflow] API format 감지 -> 그대로 사용")
        workflow = raw

    # CLIPTextEncode 노드 감지 (id 오름차순: positive=5, negative=6)
    clip_nodes = sorted(
        [nid for nid, n in workflow.items()
         if isinstance(n, dict) and n.get("class_type") == "CLIPTextEncode"],
        key=lambda x: int(x) if x.isdigit() else 0,
    )
    if len(clip_nodes) < 2:
        raise ValueError(f"CLIPTextEncode 2개 필요, 발견: {clip_nodes}")

    pos_id = POSITIVE_CLIP_NODE_ID or clip_nodes[0]
    neg_id = NEGATIVE_CLIP_NODE_ID or clip_nodes[1]

    injections = []
    workflow[pos_id]["inputs"]["text"] = positive
    injections.append(f"positive -> node[{pos_id}]")
    workflow[neg_id]["inputs"]["text"] = negative
    injections.append(f"negative -> node[{neg_id}]")

    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type", "")
        if ctype == "KSampler":
            node["inputs"]["seed"] = seed
            node["inputs"]["control_after_generate"] = "fixed"
            node["inputs"]["steps"] = steps
            injections.append(f"seed={seed} steps={steps} -> node[{nid}]")
        elif ctype == "VHS_VideoCombine":
            node["inputs"]["frame_rate"] = frame_rate
            node["inputs"]["pingpong"]   = pingpong
            injections.append(f"frame_rate={frame_rate} pingpong={pingpong} -> node[{nid}]")
        elif ctype == "LoadImage":
            node["inputs"]["image"] = image_filename
            injections.append(f"image={image_filename} -> node[{nid}]")
        elif ctype == "WanImageToVideo":
            node["inputs"]["width"]  = width
            node["inputs"]["height"] = height
            injections.append(f"width={width} height={height} -> node[{nid}]")

    logger.info(f"[Workflow] 주입 완료: {injections}")
    return workflow



def _apply_post_compositing(
    video_path:      str,
    orig_image_path: str,
    mask_path:       str,
) -> None:
    """
    생성된 비디오 위에 원본 이미지를 마스크 기반으로 합성 후 재인코딩.

    마스크 규칙 (wan_mask_generator.py 기준):
        흰색(255) = fixed zone  → 원본 이미지 픽셀 강제 덮어쓰기
        검정(0)   = moving zone → 생성된 비디오 픽셀 유지

    SetLatentNoiseMask가 WAN video latent(5D)와 비호환이므로
    Latent 마스킹 대신 픽셀 레벨 후처리로 대체.
    """
    import glob as _glob
    import shutil as _shutil
    import subprocess as _sp
    import tempfile as _tf

    from PIL import Image as _Image

    with _tf.TemporaryDirectory() as tmpdir:
        # 1. 비디오 → 프레임 분할
        _sp.run(
            ["ffmpeg", "-i", video_path,
             f"{tmpdir}/frame_%04d.png",
             "-y", "-loglevel", "quiet"],
            check=True,
        )

        frames = sorted(_glob.glob(f"{tmpdir}/frame_*.png"))
        if not frames:
            raise RuntimeError("프레임 추출 실패")

        orig_img = _Image.open(orig_image_path).convert("RGB")
        mask_img = _Image.open(mask_path).convert("L")

        # 원본 이미지 크기를 비디오 프레임 크기에 맞춤
        frame0   = _Image.open(frames[0]).convert("RGB")
        if orig_img.size != frame0.size:
            orig_img = orig_img.resize(frame0.size, _Image.LANCZOS)
            mask_img = mask_img.resize(frame0.size, _Image.LANCZOS)

        # 2. 각 프레임 합성
        # Image.composite(image1, image2, mask):
        #   mask=255(흰색) → image1(orig_img) 사용  ← fixed zone
        #   mask=0(검정)   → image2(frame_img) 사용 ← moving zone
        for f_path in frames:
            frame_img = _Image.open(f_path).convert("RGB")
            comp = _Image.composite(orig_img, frame_img, mask_img)
            comp.save(f_path)

        # 3. 원본 fps 파싱
        fps_raw = _sp.run(
            ["ffprobe", "-v", "error", "-select_streams", "v",
             "-of", "default=noprint_wrappers=1:nokey=1",
             "-show_entries", "stream=r_frame_rate", video_path],
            capture_output=True, text=True,
        ).stdout.strip()
        try:
            num, den = fps_raw.split("/")
            fps_str = str(round(int(num) / int(den)))
        except Exception:
            fps_str = "16"

        # 4. 합성 프레임 → 비디오 재인코딩
        out_tmp = f"{tmpdir}/out.mp4"
        _sp.run(
            ["ffmpeg", "-framerate", fps_str,
             "-i", f"{tmpdir}/frame_%04d.png",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-y", "-loglevel", "quiet", out_tmp],
            check=True,
        )
        _shutil.move(out_tmp, video_path)


def _inject_mask_into_workflow(workflow: dict, mask_filename: str) -> dict:
    """
    워크플로우에 마스킹 노드 주입.

    추가 노드:
        98 : LoadImage         — 마스크 이미지 로드
        99 : ImageToMask       — L채널 추출 → MASK 타입 변환
        100: SetLatentNoiseMask — KSampler의 latent_image에 마스크 주입

    KSampler의 latent_image 입력을 WanImageToVideo 직결에서
    SetLatentNoiseMask 경유로 변경.

    마스크 규칙:
        흰색(1.0) = noise 정상 적용 (moving zone)
        검정(0.0) = noise 억제 (fixed zone → 원본 픽셀 유지)
    """
    import copy
    wf = copy.deepcopy(workflow)

    # KSampler 노드 찾기
    ksampler_id = None
    wan_i2v_id  = None
    for nid, node in wf.items():
        if isinstance(node, dict):
            if node.get("class_type") == "KSampler":
                ksampler_id = nid
            elif node.get("class_type") == "WanImageToVideo":
                wan_i2v_id = nid

    if ksampler_id is None or wan_i2v_id is None:
        logger.warning("[MaskInject] KSampler 또는 WanImageToVideo 노드 없음 → 마스킹 건너뜀")
        return wf

    # 새 노드 ID (기존 ID와 충돌 방지: 기존 최대 ID + 1부터)
    existing_ids = [int(k) for k in wf.keys() if k.isdigit()]
    next_id = max(existing_ids) + 1
    load_mask_id    = str(next_id)
    img2mask_id     = str(next_id + 1)
    set_noise_id    = str(next_id + 2)

    # 노드 98: LoadImage (마스크 이미지 로드)
    wf[load_mask_id] = {
        "class_type": "LoadImage",
        "_meta": {"title": "마스크 이미지 로드"},
        "inputs": {
            "image":  mask_filename,
            "upload": "image",
        },
    }

    # 노드 99: ImageToMask (RGB → MASK, red 채널 사용)
    wf[img2mask_id] = {
        "class_type": "ImageToMask",
        "_meta": {"title": "이미지 → 마스크 변환"},
        "inputs": {
            "image":   [load_mask_id, 0],
            "channel": "red",
        },
    }

    # 노드 100: SetLatentNoiseMask (latent + mask → masked latent)
    # WanImageToVideo의 latent 출력(슬롯 2)을 마스크와 결합
    wf[set_noise_id] = {
        "class_type": "SetLatentNoiseMask",
        "_meta": {"title": "Latent 노이즈 마스크 적용"},
        "inputs": {
            "samples": [wan_i2v_id, 2],   # WanImageToVideo latent 출력
            "mask":    [img2mask_id, 0],
        },
    }

    # KSampler의 latent_image 입력을 SetLatentNoiseMask 출력으로 교체
    wf[ksampler_id]["inputs"]["latent_image"] = [set_noise_id, 0]

    logger.info(
        f"[MaskInject] 마스킹 주입 완료: "
        f"LoadImage[{load_mask_id}] → ImageToMask[{img2mask_id}] "
        f"→ SetLatentNoiseMask[{set_noise_id}] → KSampler[{ksampler_id}]"
    )
    return wf


def build_wan_workflow(
    image_filename: str,
    positive:       str,
    negative:       str,
    frame_rate:     int,
    seed:           int,
    width:          int = 480,
    height:         int = 480,
    frames:         int = 33,
    steps:          int = 20,
    cfg:            float = 7.0,
    pingpong:       bool = False,
) -> dict:
    """
    WAN I2V ComfyUI 워크플로우 JSON 반환.

    우선순위:
        1. WAN_WORKFLOW_PATH 파일이 존재하면 → 파일 로드 + 파라미터 주입
        2. 파일 없으면 → 코드 기반 빌드 (하드코딩 fallback)

    파일 로드 방식:
        wan21_native_i2v_1.json 을 그대로 사용.
        seed / frame_rate / image / positive / negative 만 주입.
        나머지 설정(steps, cfg, sampler 등)은 파일 값 유지.

    코드 빌드 방식 (fallback):
        wan21_native_i2v_1 구조 기반 하드코딩.
        노드 구성:
            2  : CLIPLoader        (umt5_xxl_fp8_e4m3fn_scaled.safetensors)
            3  : VAELoader         (wan_2.1_vae.safetensors)
            4  : CLIPVisionLoader  (clip_vision_h.safetensors)
            5  : CLIPTextEncode    (포지티브)
            6  : CLIPTextEncode    (네거티브)
            7  : LoadImage         (입력 이미지)
            8  : CLIPVisionEncode
            9  : WanImageToVideo
            10 : KSampler
            11 : VAEDecode
            12 : VHS_VideoCombine
            16 : UnetLoaderGGUF    (wan2.1-i2v-14b-480p-Q3_K_S.gguf)
    """
    # ── 파일 로드 방식 (우선) ──────────────────────────────────
    if WAN_WORKFLOW_PATH and os.path.isfile(WAN_WORKFLOW_PATH):
        logger.info(f"[Workflow] 파일 로드: {WAN_WORKFLOW_PATH}")
        return load_workflow_from_file(
            workflow_path  = WAN_WORKFLOW_PATH,
            image_filename = image_filename,
            positive       = positive,
            negative       = negative,
            frame_rate     = frame_rate,
            seed           = seed,
            width          = width,
            height         = height,
            steps          = steps,
            pingpong       = pingpong,
        )

    # ── 코드 빌드 방식 (fallback) ─────────────────────────────
    logger.info(
        f"[Workflow] 파일 없음 ({WAN_WORKFLOW_PATH}) → 코드 빌드 방식 사용"
    )
    return {
        "2": {
            "class_type": "CLIPLoader",
            "_meta": {"title": "CLIP 로드"},
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type":      "wan",
                "device":    "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "_meta": {"title": "VAE 로드"},
            "inputs": {"vae_name": VAE_MODEL},
        },
        "4": {
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "CLIP_VISION 로드"},
            "inputs": {"clip_name": CLIP_MODEL},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP 텍스트 인코딩 (포지티브)"},
            "inputs": {
                "clip": ["2", 0],
                "text": positive,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP 텍스트 인코딩 (네거티브)"},
            "inputs": {
                "clip": ["2", 0],
                "text": negative,
            },
        },
        "7": {
            "class_type": "LoadImage",
            "_meta": {"title": "이미지 로드"},
            "inputs": {
                "image":  image_filename,
                "upload": "image",
            },
        },
        "8": {
            "class_type": "CLIPVisionEncode",
            "_meta": {"title": "CLIP_VISION 인코딩"},
            "inputs": {
                "clip_vision": ["4", 0],
                "image":       ["7", 0],
                "crop":        "center",
            },
        },
        "9": {
            "class_type": "WanImageToVideo",
            "_meta": {"title": "WAN 비디오 생성 (이미지 → 비디오)"},
            "inputs": {
                "positive":           ["5", 0],
                "negative":           ["6", 0],
                "vae":                ["3", 0],
                "clip_vision_output": ["8", 0],
                "start_image":        ["7", 0],
                "width":              width,
                "height":             height,
                "length":             frames,
                "batch_size":         1,
            },
        },
        "10": {
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
            "inputs": {
                "model":        ["16", 0],
                "positive":     ["9", 0],
                "negative":     ["9", 1],
                "latent_image": ["9", 2],
                "seed":         seed,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": "euler_ancestral",
                "scheduler":    "normal",
                "denoise":      1.0,
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE 디코딩"},
            "inputs": {
                "samples": ["10", 0],
                "vae":     ["3", 0],
            },
        },
        "12": {
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine"},
            "inputs": {
                "images":          ["11", 0],
                "frame_rate":      frame_rate,
                "loop_count":      0,
                "filename_prefix": "video/h264-mp4",
                "format":          "video/h264-mp4",
                "pix_fmt":         "yuv420p",
                "crf":             19,
                "save_metadata":   True,
                "pingpong":        pingpong,
                "save_output":     True,
                "trim_to_audio":   False,
            },
        },
        "16": {
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "WAN GGUF 모델 로드"},
            "inputs": {"unet_name": WAN_MODEL},
        },
    }


# ─────────────────────────────────────────────────────────────
# WAN 백엔드 메인
# ─────────────────────────────────────────────────────────────

class WanBackend:
    """
    WAN I2V 자동화 파이프라인 메인 클래스.

    사용법:
        backend = WanBackend(api_key="sk-ant-...")
        result  = backend.generate("frog.png", output_dir="outputs/")

        if result.success:
            print(f"생성 완료: {result.video_path}")
            print(f"액션: {result.analysis.action_desc}")
            print(f"시도 횟수: {result.attempts}")
    """

    def __init__(
        self,
        api_key:       str,
        comfyui_url:   str = COMFYUI_URL,
        output_dir:    str = "outputs/wan",
        workflow_path: str | None = None,
    ) -> None:
        self.analyzer      = WanVisionAnalyzer(api_key=api_key)
        self.validator     = WanValidator()
        self.ai_validator  = WanAIValidator(api_key=api_key)
        self.bg_remover    = WanBgRemover()
        self.mask_gen      = WanMaskGenerator()
        self.mode_classifier = WanModeClassifier(api_key=api_key)
        self.post_motion_classifier = WanPostMotionClassifier(api_key=api_key)
        self.keyframe_generator = WanKeyframeGenerator()
        self.comfyui       = ComfyUIClient(base_url=comfyui_url)
        self.output_dir   = output_dir
        # 출력 하위 디렉토리 분리
        self.dir_motion          = os.path.join(output_dir, "motion")
        self.dir_keyframe_only   = os.path.join(output_dir, "keyframe_only")
        self.dir_motion_keyframe = os.path.join(output_dir, "motion_keyframe")
        for d in [output_dir, self.dir_motion, self.dir_keyframe_only, self.dir_motion_keyframe]:
            os.makedirs(d, exist_ok=True)

        # ── 검증 실패 통계 추적기 ──────────────────────────────
        self._stats = _ValidationStats(output_dir=output_dir)

        # 워크플로우 경로 결정 (인자 > 환경변수 > 기본 경로 순)
        global WAN_WORKFLOW_PATH
        if workflow_path:
            WAN_WORKFLOW_PATH = workflow_path

        if WAN_WORKFLOW_PATH and os.path.isfile(WAN_WORKFLOW_PATH):
            logger.info(f"[WanBackend] 워크플로우 파일 사용: {WAN_WORKFLOW_PATH}")
        else:
            logger.warning(
                f"[WanBackend] 워크플로우 파일 없음 ({WAN_WORKFLOW_PATH}) "
                f"→ 코드 빌드 방식 사용"
            )

    def generate(
        self,
        image_path:  str,
        output_dir:  str | None = None,
    ) -> WanResult:
        """
        이미지를 입력받아 WAN I2V 애니메이션 자동 생성.

        Args:
            image_path : 입력 이미지 경로
            output_dir : 결과 저장 디렉토리 (None이면 self.output_dir 사용)

        Returns:
            WanResult
        """
        stem = Path(image_path).stem

        # ── Stage 1: 처리 모드 분류 ─────────────────────────
        # 이미지를 분석하여 WAN이 필요한지 사전 판단
        # KEYFRAME_ONLY → WAN 생성 건너뜀 (GPU 비용 절감)
        # MOTION_NEEDED → WAN 생성 진행 → Stage 2에서 키프레임 여부 재판단
        logger.info(f"\n{'═'*55}")
        logger.info(f"  [WanBackend] 시작: {stem}")
        logger.info(f"{'═'*55}")

        mode = self.mode_classifier.classify(image_path)
        logger.info(f"  [Stage1] mode={mode.processing_mode.value} "
                    f"facing={mode.facing_direction.value} "
                    f"scene={mode.is_scene} "
                    f"action={mode.suggested_action}")

        if mode.processing_mode == ProcessingMode.KEYFRAME_ONLY:
            # 키프레임 생성 → keyframe_only/ 폴더에 저장
            kf_config = self.keyframe_generator.generate(
                suggested_action = mode.suggested_action,
                facing_direction = mode.facing_direction.value,
            )
            kf_path = os.path.join(self.dir_keyframe_only, f"{stem}_keyframe.json")
            kf_config.to_json(kf_path)

            logger.info(
                f"\n  ✅ KEYFRAME_ONLY 판정 → WAN 생성 건너뜀\n"
                f"  → subject: {mode.subject_desc}\n"
                f"  → facing:  {mode.facing_direction.value}\n"
                f"  → action:  {mode.suggested_action}\n"
                f"  → keyframe: {kf_config.animation_type} "
                f"{kf_config.duration_ms}ms {len(kf_config.keyframes)}kf\n"
                f"  → saved:   {kf_path}\n"
                f"  → reason:  {mode.reason}\n"
                f"{'═'*55}"
            )
            return WanResult(
                success         = True,
                mode            = mode,
                keyframe_config = kf_config,
            )

        # MOTION_NEEDED → WAN 생성 진행
        logger.info(f"  → MOTION_NEEDED: WAN 모션 생성 진행 (Stage 2에서 키프레임 여부 재판단)")

        # 모션 출력은 motion/ 하위 폴더에 저장
        out_dir = output_dir or self.dir_motion
        os.makedirs(out_dir, exist_ok=True)

        # ── Step 0: 이미지 전처리 (480×480 패딩) ─────────────
        # WAN 입력 / 마스크 / 후처리 합성이 동일 좌표계를 공유하도록 강제
        processed_path = os.path.join(out_dir, f"{stem}_processed.png")
        preprocess_image_simple(image_path, processed_path)
        image_path = processed_path  # 이후 모든 로직은 480×480 기준

        # ── Step 1: LLM Vision 분석 ──────────────────────────

        analysis = self.analyzer.analyze(image_path)  # processed 기준으로 분석

        logger.info(f"  → 액션: {analysis.action_desc}")
        logger.info(f"  → 오브젝트: {analysis.object_desc}")
        logger.info(f"  → 움직이는 부위: {analysis.moving_parts}")
        logger.info(f"  → 고정 부위: {analysis.fixed_parts}")
        logger.info(f"  → 근거: {analysis.reason}")
        logger.info(f"  → fps={analysis.frame_rate} "
                    f"motion={analysis.min_motion}~{analysis.max_motion}")
        logger.info(f"  → pingpong={analysis.pingpong} "
                    f"bg_type={analysis.bg_type} bg_remove={analysis.bg_remove}")

        # ── Step 2: 이미지 업로드 ──────────────────────────────────────────
        uploaded_name = self.comfyui.upload_image(image_path)

        # ── Step 2.5: 마스크 생성 (로컬 경로 보존, ComfyUI 업로드 불필요) ──
        # SetLatentNoiseMask는 WAN video latent(5D)와 비호환 → 후처리 합성 방식 사용
        # 마스크는 생성 후 파이썬 레벨에서 픽셀 덮어씌우기에만 사용
        mask_name: str | None = None       # _generate_one 시그니처 호환용 (미사용)
        mask_path_local: str | None = None
        try:
            import tempfile as _tempfile
            mask_path_local = self.mask_gen.generate(
                image_path  = image_path,
                moving_zone = analysis.moving_zone,
                output_dir  = _tempfile.mkdtemp(),
            )
            logger.info(
                f"  → 마스크 생성 완료 (후처리용): {mask_path_local} "
                f"(zone={analysis.moving_zone})"
            )
        except Exception as e:
            logger.warning(f"  ⚠ 마스크 생성 실패 (합성 없이 진행): {e}")
            mask_path_local = None

        # ── Step 3: 생성 + 검증 루프 ─────────────────────────
        # LLM이 결정한 수치로 동적 파라미터 구성
        current_fps         = analysis.frame_rate
        current_frame_count = analysis.frame_count
        current_scale       = 0.65

        # 배경 보호 문구: bg_type=solid일 때만 추가
        # bg_type=scene이면 배경이 장면이므로 흰색 강제 문구 불필요
        if analysis.bg_type == "solid":
            BG_POSITIVE = (
                "纯白色背景，整个动画过程中背景始终保持白色，背景干净无杂质，"
                "始终保持恒定的亮度，没有闪烁，画面明亮清晰，边缘锐利，无残影"
            )
            BG_NEGATIVE = (
                "背景变色，背景变暗，背景变黑，背景变灰，背景变黄，"  # noqa: RUF001
                "背景变紫，背景颜色偏移，非白色背景，"
                "黑屏，阴影遮盖，滤镜感，曝光不足，画面闪烁"
            )
        else:
            # scene 배경: 배경 유지 문구로 대체
            BG_POSITIVE = (
                "背景保持不变，整个动画过程中背景始终保持原始状态，"
                "画面明亮清晰，边缘锐利，无残影"
            )
            BG_NEGATIVE = (
                "背景消失，背景模糊，背景扭曲，"
                "黑屏，画面闪烁"
            )
        # 베이스 프롬프트: 액션 전환 시에만 교체, 중간에 누적하지 않음
        base_positive = BG_POSITIVE + ", " + analysis.positive
        base_negative = analysis.negative + ", " + BG_NEGATIVE

        # ── 이전 실패 이력 기반 negative 강화 ─────────────────
        # 동일 이미지의 이전 실패 기록이 있으면, 빈발 이슈를 negative에 자동 추가
        ISSUE_NEGATIVE_MAP = {
            "ghosting":               "残影，鬼影，半透明残像，画面闪烁，细节闪烁",
            "unnatural_movement":     "身体变形，身体拉伸，身体扭曲，动作不自然，轨迹突变",
            "character_inconsistency":"风格改变，纹理重建，角色外观变化，颜色失真",
            "no_motion":              "",  # positive 강화 대상 (negative 아님)
            "too_slow":               "",  # positive 강화 대상 (negative 아님)
            "background_color_change":"背景变色，背景变暗，背景变灰",
            "frame_escape":           "画面外移动，超出边界",
            "no_return_to_origin":    "动作不回归，姿势偏移",
            "speed_too_fast":         "动作过快，快速移动，急速运动",
            "speed_too_slow":         "",  # positive 강화 대상
            "flickering":             "画面闪烁，亮度变化，闪烁不定",
            "repeated_motion":        "重复动作，动作循环不自然",
        }
        history = self._stats.load_history(stem)
        if history["total_attempts"] > 0:
            # 이전 이력에서 이슈 빈도 집계
            issue_counter: Counter = Counter()
            for img_data in history["images"].values():
                for a in img_data["attempts"]:
                    issue_counter.update(a.get("issues", []))
                    issue_counter.update(a.get("ai_issues", []))

            # 빈도 2회 이상인 이슈만 negative에 추가
            history_negatives = []
            for issue, count in issue_counter.most_common():
                if count >= 2 and ISSUE_NEGATIVE_MAP.get(issue, ""):
                    history_negatives.append(ISSUE_NEGATIVE_MAP[issue])

            if history_negatives:
                history_neg_str = "，".join(history_negatives)
                base_negative = base_negative + ", " + history_neg_str
                logger.info(
                    f"  [이력 강화] 이전 {history['total_attempts']}회 실패 기반 "
                    f"negative 추가: {history_neg_str[:80]}..."
                )
            else:
                logger.info(
                    f"  [이력] 이전 {history['total_attempts']}회 기록 있음 "
                    f"(빈도 2회 이상 이슈 없음 → 강화 스킵)"
                )
        # ──────────────────────────────────────────────────────
        # 조정 프롬프트: 매 시도 후 실패 원인에 맞게 교체 (누적 금지)
        adj_positive  = ""
        adj_negative  = ""

        def build_prompts():
            pos = base_positive + (", " + adj_positive if adj_positive else "")
            neg = base_negative + (", " + adj_negative if adj_negative else "")
            return pos, neg

        current_positive, current_negative = build_prompts()
        logger.info(f"  → positive: {current_positive[:120]}")
        logger.info(f"  → negative: {current_negative[:100]}")

        # 성공 결과 누적 리스트 (MAX_RETRIES 횟수 모두 생성 후 반환)
        successful_results: list[WanResult] = []

        # 동일 AI 이슈 연속 발생 카운터
        # unnatural_movement 또는 character_inconsistency가 N회 연속이면
        # 프롬프트 수정이 아닌 액션 전환을 AI에 요청
        CONSECUTIVE_FAIL_THRESHOLD = 2
        consecutive_quality_fails  = 0   # unnatural/character 연속 카운트
        consecutive_nomotion_fails = 0   # no_motion/too_slow 연속 카운트
        QUALITY_ISSUES = {"unnatural_movement", "character_inconsistency"}

        for attempt in range(1, MAX_RETRIES + 1):
            seed = random.randint(0, 2**32 - 1)
            # VRAM 사용량 측정 (생성 전)
            vram_pre = self.comfyui._get_vram_usage()
            vram_used = vram_pre['used'] if vram_pre else None
            vram_str = f"VRAM={vram_pre['used']}MB/{vram_pre['total']}MB" if vram_pre else ""
            logger.info(
                f"\n  [시도 {attempt}/{MAX_RETRIES}] "
                f"seed={seed} fps={current_fps}"
                f" {vram_str}"
            )

            try:
                video_path = self._generate_one(
                    uploaded_name = uploaded_name,
                    positive      = current_positive,
                    negative      = current_negative,
                    frame_rate    = current_fps,
                    frame_count   = current_frame_count,
                    seed          = seed,
                    out_dir       = out_dir,
                    stem          = stem,
                    attempt       = attempt,
                    pingpong      = analysis.pingpong,
                    # mask_name 미전달: SetLatentNoiseMask는 WAN video latent와 비호환
                )
            except Exception as e:
                logger.warning(f"  [시도 {attempt}] 생성 실패: {e}")
                continue

            # ── Step 4: 검증 (WAN 원본 영상 기준) ───────────────
            validation = self.validator.validate(video_path, analysis=analysis)

            if validation.passed:
                logger.info(f"  ✅ 수치 검증 통과 → AI 검증 시작")

                # ── AI 검증 (수치 통과 후 추가 검증) ────────
                ai_result = self.ai_validator.validate(
                    video_path    = video_path,
                    original_path = image_path,
                    current_fps   = current_fps,
                    current_scale = current_scale,
                    positive      = current_positive,
                    negative      = current_negative,
                )
                logger.info(f"  [AIValidator] {ai_result}")

                # 수치 검증 통과 후 AI 판정 완화 처리
                SOFT_ISSUES = {"background_color_change"}
                if not ai_result.passed and ai_result.issues:
                    hard_issues = [i for i in ai_result.issues if i not in SOFT_ISSUES]
                    if not hard_issues:
                        logger.info(
                            f"  ⚠ AI soft 판정만 있음 {ai_result.issues} "
                            f"→ 수치 검증 통과했으므로 PASS 처리"
                        )
                        ai_result.passed = True
                        self._stats.record(stem, "success", [], remedy="soft_pass",
                                           remedy_detail=f"soft_issues={ai_result.issues}",
                                           vram_mb=vram_used)

                if ai_result.passed:
                    # soft_pass에서 이미 기록한 경우가 아니면 성공 기록
                    last_rec = (self._stats._current_records[-1]
                                if self._stats._current_records else None)
                    if not (last_rec and last_rec.get("remedy") == "soft_pass"):
                        self._stats.record(stem, "success", [], vram_mb=vram_used)
                    logger.info(f"\n  ✅ AI 검증 통과! (시도 {attempt}회)")
                    logger.info(f"  → {video_path}")

                    # ── Step 4.5: 후처리 합성 (검증 통과 후 적용) ──
                    # 수치/AI 검증은 WAN 원본 기준으로 통과
                    # 합성은 최종 결과물에만 적용 (몸통 고정)
                    if mask_path_local:
                        try:
                            _apply_post_compositing(video_path, image_path, mask_path_local)
                            logger.info("  [Compositing] 후처리 합성 완료")
                        except Exception as e:
                            logger.warning(f"  ⚠ 후처리 합성 실패 (원본 영상 유지): {e}")

                    # ── 배경 제거 후처리 (성공 시 + bg_remove=True인 경우만) ──
                    if analysis.bg_remove:
                        try:
                            transparent_dir = os.path.join(
                                out_dir, f"{stem}_transparent"
                            )
                            self.bg_remover.remove_background(
                                video_path  = video_path,
                                output_dir  = transparent_dir,
                                output_apng = True,
                                output_webm = True,
                                fps         = current_fps,
                            )
                            logger.info(f"  → 배경 제거 완료: {transparent_dir}")
                        except Exception as e:
                            logger.warning(f"  ⚠ 배경 제거 실패 (무시): {e}")
                    else:
                        logger.info(f"  → 배경 제거 스킵 (bg_type={analysis.bg_type})")

                    # 성공 결과 저장 후 루프 계속 (MAX_RETRIES 전부 소진)
                    # Stage 2(키프레임 이동 판단)는 사용자가 영상 확인 후
                    # classify_post_motion() 수동 호출로 진행
                    result = WanResult(
                        success    = True,
                        video_path = video_path,
                        analysis   = analysis,
                        validation = validation,
                        attempts   = attempt,
                        seed       = seed,
                        mode       = mode,
                    )
                    successful_results.append(result)
                    logger.info(
                        f"  [저장] 성공 결과 누적: {len(successful_results)}개"
                        f" — 나머지 {MAX_RETRIES - attempt}회 계속 생성"
                    )
                    # 프롬프트/파라미터는 그대로 유지하며 다음 시도 진행
                    continue
                else:
                    # AI 검증 실패 기록
                    self._stats.record(stem, "ai_fail", ai_result.issues or [], vram_mb=vram_used)
                    logger.info(f"  ❌ AI 검증 실패: {ai_result.issues}")
                    logger.info(f"  [보존] 실패 영상: {video_path}")

                    # ── 연속 품질 실패 카운터 ─────────────────
                    # unnatural_movement / character_inconsistency가
                    # 반복되면 프롬프트 수정이 아닌 Vision 재분석으로 액션 전환
                    if ai_result.issues and set(ai_result.issues) & QUALITY_ISSUES:
                        consecutive_quality_fails += 1
                        logger.info(
                            f"  [품질실패] {consecutive_quality_fails}회 연속 "
                            f"(임계={CONSECUTIVE_FAIL_THRESHOLD})"
                        )
                    else:
                        consecutive_quality_fails = 0

                    if consecutive_quality_fails >= CONSECUTIVE_FAIL_THRESHOLD:
                        logger.info(
                            f"  ⚠ 품질 이슈 {consecutive_quality_fails}회 연속 → "
                            f"Vision 재분석으로 액션 전환"
                        )
                        consecutive_quality_fails = 0
                        # 새 이미지 분석으로 다른 액션 선택
                        # 현재 실패한 action_desc를 힌트로 전달하여 다른 것 선택 유도
                        try:
                            new_analysis = self.analyzer.analyze_with_exclusion(
                                image_path   = image_path,
                                exclude_action = analysis.action_desc,
                            )
                            analysis             = new_analysis
                            current_fps          = new_analysis.frame_rate
                            current_frame_count  = new_analysis.frame_count
                            base_positive        = BG_POSITIVE + ", " + analysis.positive
                            base_negative        = new_analysis.negative + ", " + BG_NEGATIVE
                            adj_positive         = ""
                            adj_negative         = ""
                            current_positive, current_negative = build_prompts()
                            logger.info(
                                f"  [액션전환] 새 액션: {new_analysis.action_desc} "
                                f"| moving={new_analysis.moving_parts} "
                                f"| frames={new_analysis.frame_count}"
                            )
                            self._stats._update_last_remedy(
                                "action_switch", f"→ {new_analysis.action_desc[:30]}"
                            )
                            # 마스크도 새 moving_zone 기준으로 재생성
                            if mask_path_local:
                                try:
                                    import tempfile as _tempfile
                                    mask_path_local = self.mask_gen.generate(
                                        image_path  = image_path,
                                        moving_zone = new_analysis.moving_zone,
                                        output_dir  = _tempfile.mkdtemp(),
                                    )
                                except Exception as me:
                                    logger.warning(f"  ⚠ 마스크 재생성 실패: {me}")
                        except Exception as e:
                            logger.warning(f"  ⚠ 액션 전환 분석 실패: {e}")
                        continue  # 프롬프트 수정 없이 새 액션으로 바로 재시도

                    # AI가 결정한 수치 적용
                    adjust_details = []
                    if ai_result.frame_rate is not None:
                        logger.info(f"  [AI조정] fps: {current_fps} → {ai_result.frame_rate}")
                        adjust_details.append(f"fps:{current_fps}→{ai_result.frame_rate}")
                        current_fps = ai_result.frame_rate
                    if ai_result.scale is not None:
                        logger.info(f"  [AI조정] scale: {current_scale} → {ai_result.scale}")
                        adjust_details.append(f"scale:{current_scale}→{ai_result.scale}")
                        current_scale = ai_result.scale
                    if ai_result.positive:
                        logger.info(f"  [AI조정] positive 교체: {ai_result.positive}")
                        adjust_details.append(f"pos:{ai_result.positive[:20]}")
                        adj_positive = ai_result.positive  # 누적 없이 교체
                    if ai_result.negative:
                        logger.info(f"  [AI조정] negative 교체: {ai_result.negative}")
                        adjust_details.append(f"neg:{ai_result.negative[:20]}")
                        adj_negative = ai_result.negative  # 누적 없이 교체
                    if adjust_details:
                        self._stats._update_last_remedy("ai_adjust", ", ".join(adjust_details))
                    current_positive, current_negative = build_prompts()
            else:
                # 수치 검증 실패 기록
                self._stats.record(stem, "metric_fail", validation.failed_checks, vram_mb=vram_used)
                logger.info(
                    f"  ❌ 수치 검증 실패: {validation.failed_checks}"
                )
                logger.info(f"  [보존] 실패 영상: {video_path}")

                # ── 수치 FAIL 원인 분류 ───────────────────────
                # no_motion 단독(+too_slow 조합 포함) 첫 attempt
                # — WAN이 아무것도 생성 못 한 케이스
                # 프롬프트 변경보다 seed 교체가 효과적이므로 AI 조정 없이 재시도
                MOTION_ONLY = {"no_motion", "too_slow"}
                if set(validation.failed_checks) <= MOTION_ONLY:
                    consecutive_nomotion_fails += 1
                    logger.info(
                        f"  ⚠ no_motion/too_slow {consecutive_nomotion_fails}회 연속"
                    )
                    if attempt == 1 or consecutive_nomotion_fails < CONSECUTIVE_FAIL_THRESHOLD:
                        logger.info("  → seed 교체 후 재시도 (프롬프트 유지)")
                        self._stats._update_last_remedy("seed_retry", "프롬프트 유지, seed만 교체")
                        continue
                    # 3회 연속 no_motion → 동작 자체가 WAN에 너무 어려운 것
                    # 액션 전환
                    logger.info(
                        f"  ⚠ no_motion {consecutive_nomotion_fails}회 연속 → "
                        f"Vision 재분석으로 액션 전환"
                    )
                    consecutive_nomotion_fails = 0
                    try:
                        new_analysis = self.analyzer.analyze_with_exclusion(
                            image_path     = image_path,
                            exclude_action = analysis.action_desc,
                        )
                        analysis             = new_analysis
                        current_fps          = new_analysis.frame_rate
                        current_frame_count  = new_analysis.frame_count
                        base_positive        = BG_POSITIVE + ", " + analysis.positive
                        base_negative        = new_analysis.negative + ", " + BG_NEGATIVE
                        adj_positive         = ""
                        adj_negative         = ""
                        current_positive, current_negative = build_prompts()
                        logger.info(
                            f"  [액션전환] 새 액션: {new_analysis.action_desc} "
                            f"| moving={new_analysis.moving_parts} "
                            f"| frames={new_analysis.frame_count}"
                        )
                        self._stats._update_last_remedy(
                            "action_switch", f"→ {new_analysis.action_desc[:30]}"
                        )
                    except Exception as e:
                        logger.warning(f"  ⚠ 액션 전환 실패: {e}")
                    continue
                else:
                    consecutive_nomotion_fails = 0

                # ── AI 검증 + 조정 ───────────────────────────
                ai_result = self.ai_validator.validate(
                    video_path    = video_path,
                    original_path = image_path,
                    current_fps   = current_fps,
                    current_scale = current_scale,
                    positive      = current_positive,
                    negative      = current_negative,
                )
                logger.info(f"  [AIValidator] {ai_result}")

                # 수치 실패 기록에 AI 이슈도 추가
                if ai_result.issues:
                    self._stats._update_last_ai_issues(ai_result.issues)

                # ── 연속 품질 실패 카운터 (수치 FAIL 경로) ──
                if ai_result.issues and set(ai_result.issues) & QUALITY_ISSUES:
                    consecutive_quality_fails += 1
                    logger.info(
                        f"  [품질실패] {consecutive_quality_fails}회 연속 "
                        f"(임계={CONSECUTIVE_FAIL_THRESHOLD})"
                    )
                    if consecutive_quality_fails >= CONSECUTIVE_FAIL_THRESHOLD:
                        logger.info(
                            f"  ⚠ 품질 이슈 {consecutive_quality_fails}회 연속 → "
                            f"Vision 재분석으로 액션 전환"
                        )
                        consecutive_quality_fails = 0
                        try:
                            new_analysis = self.analyzer.analyze_with_exclusion(
                                image_path     = image_path,
                                exclude_action = analysis.action_desc,
                            )
                            analysis             = new_analysis
                            current_fps          = new_analysis.frame_rate
                            current_frame_count  = new_analysis.frame_count
                            base_positive        = BG_POSITIVE + ", " + analysis.positive
                            base_negative        = new_analysis.negative + ", " + BG_NEGATIVE
                            adj_positive         = ""
                            adj_negative         = ""
                            current_positive, current_negative = build_prompts()
                            logger.info(
                                f"  [액션전환] 새 액션: {new_analysis.action_desc} "
                                f"| moving={new_analysis.moving_parts} "
                                f"| frames={new_analysis.frame_count}"
                            )
                            self._stats._update_last_remedy(
                                "action_switch", f"→ {new_analysis.action_desc[:30]}"
                            )
                            if mask_path_local:
                                try:
                                    import tempfile as _tempfile
                                    mask_path_local = self.mask_gen.generate(
                                        image_path  = image_path,
                                        moving_zone = new_analysis.moving_zone,
                                        output_dir  = _tempfile.mkdtemp(),
                                    )
                                except Exception as me:
                                    logger.warning(f"  ⚠ 마스크 재생성 실패: {me}")
                        except Exception as e:
                            logger.warning(f"  ⚠ 액션 전환 분석 실패: {e}")
                        continue
                else:
                    consecutive_quality_fails = 0

                # AI가 결정한 수치 적용
                adjust_details2 = []
                if ai_result.frame_rate is not None:
                    logger.info(f"  [AI조정] fps: {current_fps} → {ai_result.frame_rate}")
                    adjust_details2.append(f"fps:{current_fps}→{ai_result.frame_rate}")
                    current_fps = ai_result.frame_rate
                if ai_result.scale is not None:
                    logger.info(f"  [AI조정] scale: {current_scale} → {ai_result.scale}")
                    adjust_details2.append(f"scale:{current_scale}→{ai_result.scale}")
                    current_scale = ai_result.scale
                if ai_result.positive:
                    logger.info(f"  [AI조정] positive 교체: {ai_result.positive}")
                    adjust_details2.append(f"pos:{ai_result.positive[:20]}")
                    adj_positive = ai_result.positive  # 누적 없이 교체
                if ai_result.negative:
                    logger.info(f"  [AI조정] negative 교체: {ai_result.negative}")
                    adjust_details2.append(f"neg:{ai_result.negative[:20]}")
                    adj_negative = ai_result.negative  # 누적 없이 교체
                if adjust_details2:
                    self._stats._update_last_remedy("ai_adjust", ", ".join(adjust_details2))
                current_positive, current_negative = build_prompts()

        # ── 루프 종료: 통계 출력 + VRAM 초기화 ─────────────────
        # 현재 이미지 실패 비율 + 누적 실패 비율 출력
        self._stats.print_image_summary(stem)
        self._stats.print_cumulative_summary()
        self._stats.save_to_file()

        # VRAM 초기화: 다음 이미지 생성 전 누적 메모리 해제
        self.comfyui.free_memory()

        if successful_results:
            logger.info(
                f"\n  ✅ {MAX_RETRIES}회 완료. 성공 {len(successful_results)}개 중 첫 번째 반환."
            )
            return successful_results[0]

        # ── 전부 실패 ─────────────────────────────────────────
        logger.warning(
            f"\n  ⚠ {MAX_RETRIES}회 시도 후 전부 실패."
        )
        return WanResult(
            success  = False,
            analysis = analysis,
            attempts = MAX_RETRIES,
            mode     = mode,
        )

    def _adjust_params(
        self,
        fps:           int,
        scale:         float,
        failed_checks: list[str],
    ) -> tuple[int, float]:
        """
        검증 실패 원인에 따라 파라미터 자동 조정.

        too_fast        → frame_rate 단계적 감소
        too_slow        → frame_rate 단계적 증가
        repeated_motion → 시드 재시도 (변경 없음)
        frame_escape    → scale 감소
        """
        if "too_fast" in failed_checks:
            fps_steps = [24, 22, 20, 18, 16, 14, 12, 10]
            try:
                idx = fps_steps.index(fps)
                fps = fps_steps[min(idx + 1, len(fps_steps) - 1)]
            except ValueError:
                fps = max(10, int(fps * 0.8))
            logger.info(f"  [조정] too_fast → fps={fps}")

        if "too_slow" in failed_checks:
            fps_steps = [10, 12, 14, 16, 18, 20, 22, 24]
            try:
                idx = fps_steps.index(fps)
                fps = fps_steps[min(idx + 1, len(fps_steps) - 1)]
            except ValueError:
                fps = min(24, int(fps * 1.2))
            logger.info(f"  [조정] too_slow → fps={fps}")

        if "frame_escape" in failed_checks:
            scale = max(0.45, round(scale - 0.05, 2))
            logger.info(f"  [조정] frame_escape → scale={scale}")

        if "repeated_motion" in failed_checks:
            logger.info(f"  [조정] repeated_motion → 시드 재시도")

        return fps, scale

    # ──────────────────────────────────────────────────────────
    # Stage 2: 모션 생성 후 키프레임 이동 판단 (수동 호출)
    # ──────────────────────────────────────────────────────────

    def classify_post_motion(
        self,
        video_path:    str,
        image_path:    str | None = None,
    ) -> "PostMotionResult":
        """
        [Stage 2] 생성된 영상을 분석하여 키프레임 이동이 필요한지 판단.

        WAN 생성 완료 후, 사용자가 영상을 확인하고 원하는 영상을 지정하여 호출.
        generate() 내부에서는 자동 호출되지 않음.

        Args:
            video_path  : 분석할 MP4 영상 경로
                          (예: "outputs/wan/bird_attempt03.mp4")
            image_path  : 원본 이미지 경로 (없으면 processed 이미지 자동 추론)

        Returns:
            PostMotionResult:
                needs_keyframe   : bool  — 키프레임 이동 필요 여부
                travel_type      : str   — no_travel / travel_lateral / travel_vertical / travel_diagonal
                travel_direction : str   — left / right / up / down / none
                confidence       : float — 판단 확신도 (0.0~1.0)
                reason           : str   — 판단 근거

        사용법:
            # 1. WAN 생성
            result = backend.generate("frog.png")

            # 2. outputs/wan/ 에서 생성된 영상 확인 (attempt01~07.mp4)

            # 3. 원하는 영상으로 Stage 2 호출
            pm = backend.classify_post_motion("outputs/wan/frog_attempt03.mp4")

            if pm.needs_keyframe:
                print(f"이동 필요: {pm.travel_direction.value} 방향")
            else:
                print("제자리 모션 — 키프레임 이동 불필요")
        """
        logger.info(f"\n{'═'*55}")
        logger.info(f"  [Stage2] 수동 모션 분석 시작")
        logger.info(f"  → 영상: {video_path}")
        logger.info(f"{'═'*55}")

        # 원본 이미지 경로 추론
        if image_path is None:
            stem = Path(video_path).stem
            # attempt 번호 제거: bird_attempt03 → bird
            base_stem = re.sub(r'_attempt\d+$', '', stem)
            candidate = os.path.join(
                os.path.dirname(video_path),
                f"{base_stem}_processed.png",
            )
            if os.path.exists(candidate):
                image_path = candidate
                logger.info(f"  → 원본 이미지 자동 추론: {image_path}")
            else:
                logger.warning(f"  ⚠ 원본 이미지 추론 실패: {candidate}")

        result = self.post_motion_classifier.classify(
            video_path    = video_path,
            original_path = image_path or video_path,
        )

        if result.needs_keyframe:
            logger.info(
                f"\n  ✅ 키프레임 이동 필요\n"
                f"  → type:      {result.travel_type.value}\n"
                f"  → direction: {result.travel_direction.value}\n"
                f"  → confidence:{result.confidence:.2f}\n"
                f"  → reason:    {result.reason}\n"
                f"{'═'*55}"
            )
        else:
            logger.info(
                f"\n  ℹ 키프레임 이동 불필요 (제자리 모션)\n"
                f"  → confidence:{result.confidence:.2f}\n"
                f"  → reason:    {result.reason}\n"
                f"{'═'*55}"
            )

        return result

    # ──────────────────────────────────────────────────────────
    # 수동 선택 영상 최종 처리
    # ──────────────────────────────────────────────────────────

    def finalize_manual_selection(
        self,
        video_path:    str,
        image_path:    str | None = None,
        output_dir:    str | None = None,
    ) -> dict:
        """
        사용자가 수동으로 선택한 영상을 최종 결과물로 처리.

        자동 파이프라인이 10회 실패한 경우에도 사용 가능.
        후처리 합성(마스크가 있을 경우) → 배경 제거 → APNG 생성.
        fps는 영상 파일에서 자동 추출.

        Args:
            video_path  : 선택한 MP4 영상 경로
                          (예: "outputs/wan/bird_with_room_attempt04.mp4")
            image_path  : 원본 이미지 경로 (합성용, 없으면 합성 생략)
            output_dir  : 결과물 저장 디렉토리
                          (기본: 영상과 동일 디렉토리 내 _transparent 폴더)

        Returns:
            {
              "video_path":      str,   # 입력 영상 경로 (변경 없음)
              "transparent_dir": str,   # 투명 PNG 시퀀스 디렉토리
              "apng_path":       str,   # APNG 파일 경로
              "fps":             int,   # 영상에서 자동 추출한 fps
              "success":         bool,
              "error":           str | None,
            }
        """
        stem = Path(video_path).stem
        out_dir = output_dir or os.path.join(
            os.path.dirname(video_path), f"{stem}_transparent"
        )
        os.makedirs(out_dir, exist_ok=True)

        # ── fps 자동 추출 ──────────────────────────────────────
        fps = 16  # fallback
        try:
            import subprocess as _sp2, json as _json
            probe = _sp2.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", video_path],
                capture_output=True, text=True,
            )
            streams = _json.loads(probe.stdout).get("streams", [])
            for s in streams:
                if s.get("codec_type") == "video":
                    raw = s.get("r_frame_rate", "16/1")
                    num, den = (int(x) for x in raw.split("/"))
                    fps = num // den if den else 16
                    break
            logger.info(f"  [수동선택] 영상 fps 자동 감지: {fps}")
        except Exception:
            logger.warning("  [수동선택] fps 감지 실패 → 기본값 16 사용")

        result = {
            "video_path":      video_path,
            "transparent_dir": out_dir,
            "apng_path":       None,
            "fps":             fps,
            "success":         False,
            "error":           None,
        }

        logger.info(f"\n{'═'*55}")
        logger.info(f"  [수동선택] 최종 처리 시작: {Path(video_path).name}")
        logger.info(f"{'═'*55}")

        # ── 후처리 합성 (image_path 제공 시) ──────────────────
        # 실패 영상들은 합성 전 상태로 저장되어 있으므로
        # 수동 선택 시 합성을 적용할 수 있도록 지원
        if image_path:
            # processed 이미지 경로 추론 (preprocess 결과물)
            img_stem = Path(image_path).stem
            processed_candidate = os.path.join(
                os.path.dirname(video_path),
                f"{img_stem}_processed.png",
            )
            use_image = processed_candidate if os.path.exists(processed_candidate) else image_path

            # 마스크 경로 추론 (동일 이름 패턴으로 저장된 것 찾기)
            mask_candidate = os.path.join(
                os.path.dirname(video_path),
                f"{img_stem}_processed_mask.png",
            )
            if not os.path.exists(mask_candidate):
                # tmp 디렉토리에 있을 수 있으므로 합성 생략
                mask_candidate = None

            if mask_candidate:
                try:
                    _apply_post_compositing(video_path, use_image, mask_candidate)
                    logger.info("  [수동선택] 후처리 합성 완료")
                except Exception as e:
                    logger.warning(f"  [수동선택] 후처리 합성 실패 (원본 영상 유지): {e}")
            else:
                logger.info("  [수동선택] 마스크 없음 → 합성 생략, 영상 그대로 사용")

        # ── 배경 제거 → APNG 생성 ─────────────────────────────
        try:
            self.bg_remover.remove_background(
                video_path  = video_path,
                output_dir  = out_dir,
                output_apng = True,
                fps         = fps,
            )
            apng_path = os.path.join(out_dir, f"{stem}_transparent.apng")
            result["apng_path"] = apng_path
            result["success"]   = True
            logger.info(f"  [수동선택] 완료 → {apng_path}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"  [수동선택] 배경 제거 실패: {e}")

        return result

    def _generate_one(
        self,
        uploaded_name: str,
        positive:      str,
        negative:      str,
        frame_rate:    int,
        frame_count:   int,
        seed:          int,
        out_dir:       str,
        stem:          str,
        attempt:       int,
        pingpong:      bool = False,
        mask_name:     str | None = None,
    ) -> str:
        """단일 생성 시도. 생성된 비디오 파일 경로 반환."""
        # 워크플로우 생성 (AI 조정 수치 사용)
        workflow = build_wan_workflow(
            image_filename = uploaded_name,
            positive       = positive,
            negative       = negative,
            frame_rate     = frame_rate,
            seed           = seed,
            frames         = frame_count,
            steps          = 20,
            pingpong       = pingpong,
        )

        # 마스크 주입: SetLatentNoiseMask 노드 추가
        if mask_name:
            workflow = _inject_mask_into_workflow(workflow, mask_name)

        # 큐 추가
        prompt_id = self.comfyui.queue_prompt(workflow)

        # 완료 대기
        history = self.comfyui.wait_for_completion(prompt_id)

        # 비디오 다운로드 (ATTEMPT_OFFSET으로 기존 영상 번호 이후부터)
        actual_num = attempt + ATTEMPT_OFFSET
        video_path = os.path.join(
            out_dir,
            f"{stem}_attempt{actual_num:02d}.mp4",
        )
        return self.comfyui.download_video(history, video_path)