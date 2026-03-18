"""
image_pipeline/sprite_gen/wan_lottie_baker.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    Lottie JSON(모션)에 CSS 키프레임(translate/rotate/scale/opacity)을
    베이크하여 통합 Lottie 파일을 생성.

방식:
    래퍼 레이어(precomp) — 기존 프레임 레이어들을 precomp asset으로 이동하고,
    새 래퍼 레이어가 키프레임 transform을 Lottie 애니메이션으로 적용.
    기존 레이어를 개별 수정하지 않으므로 안전.

입력:
    - Lottie JSON 파일 (wan_lottie_converter 출력)
    - 키프레임 JSON (wan_keyframe_generator 또는 대시보드 편집 결과)

출력:
    - 통합 Lottie JSON (모션 + 키프레임)
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# CSS easing → Lottie bezier 매핑
EASING_MAP = {
    "linear":       {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "ease-in":      {"x": [0.42, 1.0], "y": [0.0, 1.0]},
    "ease-out":     {"x": [0.0, 0.58], "y": [1.0, 1.0]},
    "ease-in-out":  {"x": [0.42, 0.58], "y": [0.0, 1.0]},
}


def bake_keyframes(
    lottie_path:   str,
    keyframe_data: dict,
    output_path:   str,
) -> str:
    """
    Lottie JSON에 키프레임 transform을 베이크하여 통합 파일 생성.

    Args:
        lottie_path   : 원본 Lottie JSON 경로 (모션만)
        keyframe_data : 키프레임 dict (animation_type, keyframes, duration_ms, easing 등)
        output_path   : 통합 Lottie 출력 경로

    Returns:
        출력 파일 경로
    """
    with open(lottie_path, "r", encoding="utf-8") as f:
        lottie = json.load(f)

    lottie_combined = _apply_keyframes_to_lottie(lottie, keyframe_data)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(lottie_combined, f, separators=(",", ":"))

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(
        f"[LottieBaker] 통합 완료: {output_path} ({size_mb:.1f}MB)"
    )
    return output_path


def _apply_keyframes_to_lottie(lottie: dict, kf_data: dict) -> dict:
    """
    Lottie JSON에 키프레임을 precomp 래퍼 방식으로 주입.

    구조 변환:
        변환 전: layers = [frame_0, frame_1, ..., frame_N]
        변환 후: assets += [precomp_motion(기존 레이어들)]
                 layers = [wrapper_layer(키프레임 transform)]
    """
    result = copy.deepcopy(lottie)

    w = result.get("w", 480)
    h = result.get("h", 480)
    fps = result.get("fr", 16)
    total_frames = result.get("op", 0) - result.get("ip", 0)

    if total_frames <= 0:
        logger.warning("[LottieBaker] Lottie 프레임 없음, 원본 반환")
        return result

    keyframes = kf_data.get("keyframes", [])
    if not keyframes:
        logger.warning("[LottieBaker] 키프레임 없음, 원본 반환")
        return result

    easing_name = kf_data.get("easing", "ease-in-out")
    lottie_easing = EASING_MAP.get(easing_name, EASING_MAP["ease-in-out"])

    # ── 1. 기존 레이어 → precomp asset 이동 ──────────────────
    precomp_id = "precomp_motion"
    precomp_asset = {
        "id": precomp_id,
        "layers": result.get("layers", []),
        "fr": fps,
        "nm": "motion_layers",
    }

    if "assets" not in result:
        result["assets"] = []
    result["assets"].append(precomp_asset)

    # ── 2. 키프레임 → Lottie 애니메이션 속성 변환 ─────────────
    pos_kfs = []
    rot_kfs = []
    scale_kfs = []
    opacity_kfs = []

    cx = w / 2.0
    cy = h / 2.0

    for kf in keyframes:
        t_norm = float(kf.get("t", 0))
        frame = round(t_norm * total_frames)

        tx = float(kf.get("translateX", 0))
        ty = float(kf.get("translateY", 0))
        rot = float(kf.get("rotate", 0))
        sx = float(kf.get("scaleX", 1))
        sy = float(kf.get("scaleY", 1))
        op = float(kf.get("opacity", 1))

        # position: center + translate offset
        pos_kfs.append({
            "t": frame,
            "s": [cx + tx, cy + ty, 0],
            "e": [cx + tx, cy + ty, 0],
            "i": {"x": [lottie_easing["x"][0]], "y": [lottie_easing["y"][0]]},
            "o": {"x": [lottie_easing["x"][1]], "y": [lottie_easing["y"][1]]},
        })

        # rotation
        rot_kfs.append({
            "t": frame,
            "s": [rot],
            "e": [rot],
            "i": {"x": [lottie_easing["x"][0]], "y": [lottie_easing["y"][0]]},
            "o": {"x": [lottie_easing["x"][1]], "y": [lottie_easing["y"][1]]},
        })

        # scale (CSS 1.0 = Lottie 100)
        scale_kfs.append({
            "t": frame,
            "s": [sx * 100, sy * 100, 100],
            "e": [sx * 100, sy * 100, 100],
            "i": {"x": [lottie_easing["x"][0]], "y": [lottie_easing["y"][0]]},
            "o": {"x": [lottie_easing["x"][1]], "y": [lottie_easing["y"][1]]},
        })

        # opacity (CSS 0~1 = Lottie 0~100)
        opacity_kfs.append({
            "t": frame,
            "s": [op * 100],
            "e": [op * 100],
            "i": {"x": [lottie_easing["x"][0]], "y": [lottie_easing["y"][0]]},
            "o": {"x": [lottie_easing["x"][1]], "y": [lottie_easing["y"][1]]},
        })

    # 마지막 키프레임은 hold (보간 불필요)
    for kf_list in [pos_kfs, rot_kfs, scale_kfs, opacity_kfs]:
        if kf_list:
            last = kf_list[-1]
            last.pop("i", None)
            last.pop("o", None)
            last.pop("e", None)

    # ── 3. 래퍼 레이어 생성 ───────────────────────────────────
    wrapper_layer = {
        "ddd": 0,
        "ind": 0,
        "ty": 0,                  # precomp 참조 레이어
        "nm": "keyframe_wrapper",
        "refId": precomp_id,
        "sr": 1,
        "ks": {
            "o": {"a": 1, "k": opacity_kfs} if len(opacity_kfs) > 1 else {"a": 0, "k": 100},
            "r": {"a": 1, "k": rot_kfs} if len(rot_kfs) > 1 else {"a": 0, "k": 0},
            "p": {"a": 1, "k": pos_kfs} if len(pos_kfs) > 1 else {"a": 0, "k": [cx, cy, 0]},
            "a": {"a": 0, "k": [cx, cy, 0]},
            "s": {"a": 1, "k": scale_kfs} if len(scale_kfs) > 1 else {"a": 0, "k": [100, 100, 100]},
        },
        "ip": 0,
        "op": total_frames,
        "st": 0,
        "bm": 0,
        "w": w,
        "h": h,
    }

    # ── 4. 래퍼로 교체 ───────────────────────────────────────
    result["layers"] = [wrapper_layer]

    logger.info(
        f"[LottieBaker] precomp 생성: {len(precomp_asset['layers'])}레이어 → "
        f"래퍼 1레이어 (kf={len(keyframes)}개, easing={easing_name})"
    )
    return result