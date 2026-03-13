"""
image_pipeline/sprite_gen/wan_lottie_converter.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    투명 PNG 시퀀스를 Lottie JSON 애니메이션으로 변환.
    해상도/프레임 수 최적화로 웹 배포에 적합한 파일 크기 달성.

방식:
    래스터 내장 (base64) — 각 프레임 PNG를 Lottie assets에 내장.

용량 최적화:
    480×480 60프레임 원본 → ~4.0MB
    240×240 30프레임 (web 프리셋) → ~0.5MB

프리셋:
    original : 원본 그대로 (최대 품질)
    web      : 240px / 30프레임 (0.5~1.5MB 목표)
    web_hd   : 360px / 40프레임 (1~3MB 목표)
    mobile   : 180px / 24프레임 (0.3~0.8MB 목표)

의존성:
    - PIL (Pillow)
    - 표준 라이브러리만 사용 (외부 Lottie 라이브러리 불필요)
"""

from __future__ import annotations

import base64
import glob
import io
import json
import logging
import os
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


OPTIMIZE_PRESETS = {
    "original": {
        "max_size": None, "max_frames": None, "png_optimize": False,
        "desc": "원본 그대로 (최대 품질, 파일 큼)",
    },
    "web": {
        "max_size": 240, "max_frames": 30, "png_optimize": True,
        "desc": "웹 배포용 (0.5~1.5MB 목표)",
    },
    "web_hd": {
        "max_size": 360, "max_frames": 40, "png_optimize": True,
        "desc": "웹 고화질 (1~3MB 목표)",
    },
    "mobile": {
        "max_size": 180, "max_frames": 24, "png_optimize": True,
        "desc": "모바일용 (0.3~0.8MB 목표)",
    },
}


class WanLottieConverter:
    """
    투명 PNG 시퀀스 → Lottie JSON 변환기 (최적화 내장).

    사용법:
        converter = WanLottieConverter()

        # 웹 배포용 (240px, 30프레임 → ~0.5MB)
        converter.convert(
            png_dir="outputs/wan/frog_transparent/",
            output_path="outputs/wan/frog_transparent/frog.json",
            fps=16, preset="web",
        )

        # 원본 품질 (480px, 전프레임 → ~4MB)
        converter.convert(..., preset="original")

        # 커스텀 설정
        converter.convert(..., max_size=320, max_frames=25)
    """

    def convert(
        self,
        png_dir:     str,
        output_path: str,
        fps:         int  = 16,
        loop:        bool = True,
        preset:      str  = "original",
        max_size:    int | None = None,
        max_frames:  int | None = None,
        stem_filter: str | None = None,
    ) -> str:
        """
        디렉토리 내 PNG 시퀀스를 최적화된 Lottie JSON으로 변환.

        Args:
            png_dir     : 투명 PNG 시퀀스 디렉토리
            output_path : 출력 Lottie JSON 경로
            fps         : 프레임 레이트
            loop        : 루프 재생 여부
            preset      : "original" / "web" / "web_hd" / "mobile"
            max_size    : 장축 최대 px (프리셋 오버라이드)
            max_frames  : 최대 프레임 수 (프리셋 오버라이드)
            stem_filter : 파일명 필터

        Returns:
            출력 파일 경로
        """
        if stem_filter:
            pattern = os.path.join(png_dir, f"{stem_filter}_frame_*.png")
        else:
            pattern = os.path.join(png_dir, "*_frame_*.png")

        png_paths = sorted(glob.glob(pattern))
        if not png_paths:
            png_paths = sorted(glob.glob(os.path.join(png_dir, "*.png")))
            png_paths = [p for p in png_paths
                         if not p.endswith((".apng", "_transparent.apng"))]
        if not png_paths:
            raise FileNotFoundError(f"PNG 파일 없음: {png_dir}")

        p = OPTIMIZE_PRESETS.get(preset, OPTIMIZE_PRESETS["web"])
        return self._build_lottie(
            png_paths    = png_paths,
            output_path  = output_path,
            fps          = fps,
            loop         = loop,
            max_size     = max_size   if max_size   is not None else p["max_size"],
            max_frames   = max_frames if max_frames is not None else p["max_frames"],
            png_optimize = p.get("png_optimize", True),
        )

    def _build_lottie(
        self,
        png_paths:    list[str],
        output_path:  str,
        fps:          int,
        loop:         bool,
        max_size:     int | None,
        max_frames:   int | None,
        png_optimize: bool,
    ) -> str:
        total_frames = len(png_paths)

        # ── 프레임 스킵 (균등 간격 선택) ─────────────────────
        if max_frames and total_frames > max_frames:
            step = total_frames / max_frames
            indices = [int(i * step) for i in range(max_frames)]
            if indices[-1] != total_frames - 1:
                indices[-1] = total_frames - 1
            selected = [png_paths[i] for i in indices]
            logger.info(f"[Lottie] 프레임 스킵: {total_frames} → {len(selected)}")
        else:
            selected = png_paths

        frame_count = len(selected)

        # ── 리사이즈 계산 ────────────────────────────────────
        orig_w, orig_h = Image.open(selected[0]).size
        if max_size and max(orig_w, orig_h) > max_size:
            ratio = max_size / max(orig_w, orig_h)
            out_w = int(orig_w * ratio) // 2 * 2  # 짝수 맞춤
            out_h = int(orig_h * ratio) // 2 * 2
            resize = True
            logger.info(f"[Lottie] 리사이즈: {orig_w}×{orig_h} → {out_w}×{out_h}")
        else:
            out_w, out_h = orig_w, orig_h
            resize = False

        logger.info(
            f"[Lottie] 변환: {frame_count}프레임 {out_w}×{out_h} {fps}fps"
        )

        # ── Assets (각 프레임 base64 내장) ────────────────────
        assets = []
        for i, path in enumerate(selected):
            img = Image.open(path).convert("RGBA")
            if resize:
                img = img.resize((out_w, out_h), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=png_optimize)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            assets.append({
                "id": f"frame_{i}", "w": out_w, "h": out_h,
                "u": "", "p": f"data:image/png;base64,{b64}", "e": 1,
            })

        # ── Layers (프레임별 Image Layer) ─────────────────────
        layers = []
        for i in range(frame_count):
            layers.append({
                "ddd": 0, "ind": i, "ty": 2,
                "nm": f"frame_{i}", "refId": f"frame_{i}", "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [out_w / 2, out_h / 2, 0]},
                    "a": {"a": 0, "k": [out_w / 2, out_h / 2, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]},
                },
                "ip": i, "op": i + 1, "st": 0, "bm": 0,
            })

        # ── Lottie JSON ─────────────────────────────────────
        lottie = {
            "v": "5.7.0", "fr": fps,
            "ip": 0, "op": frame_count,
            "w": out_w, "h": out_h,
            "nm": Path(output_path).stem,
            "ddd": 0,
            "assets": assets, "layers": layers,
        }

        # ── 저장 ─────────────────────────────────────────────
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lottie, f, separators=(",", ":"))

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        dur_ms = int(frame_count / fps * 1000) if fps > 0 else 0
        logger.info(
            f"[Lottie] 완료: {output_path} "
            f"({size_mb:.1f}MB, {frame_count}f, {dur_ms}ms)"
        )
        return output_path

    def get_lottie_info(self, lottie_path: str) -> dict:
        """
        Lottie 정보 반환. 키프레임 동기화에서 duration 참조용.

        Returns:
            {"fps", "frame_count", "duration_ms", "width", "height", "file_size_mb"}
        """
        with open(lottie_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fps = data.get("fr", 16)
        fc = data.get("op", 0) - data.get("ip", 0)
        return {
            "fps": fps,
            "frame_count": fc,
            "duration_ms": int(fc / fps * 1000) if fps > 0 else 0,
            "width": data.get("w", 0),
            "height": data.get("h", 0),
            "file_size_mb": os.path.getsize(lottie_path) / (1024 * 1024),
        }