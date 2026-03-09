"""
image_pipeline/sprite_gen/wan_mask_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    WAN I2V 마스킹 파이프라인용 흑백 마스크 이미지 생성.

마스크 규칙 (SetLatentNoiseMask 기준):
    검정(0)   = moving zone  → KSampler가 자유롭게 생성 (denoise 정상 적용)
    흰색(255) = fixed zone   → 원본 픽셀 유지 (denoise 억제)

입력:
    - 원본 이미지 크기
    - moving_zone: [x1, y1, x2, y2] (0.0~1.0 상대좌표, vision_analyzer 출력)

출력:
    - 마스크 PNG 파일 (L mode, 흰색/검정)

설계:
    - 외부 모델 불필요 (PIL만 사용)
    - moving_zone 박스를 흰색으로, 나머지를 검정으로
    - 경계에 Gaussian blur 적용 → 자연스러운 전환 (경계 열화 방지)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# 경계 블러 반경: 픽셀 단위 (이미지 크기 대비 ~3%)
BLUR_RADIUS_RATIO = 0.03


class WanMaskGenerator:
    """
    moving_zone bbox → 흑백 마스크 PNG 생성.

    사용법:
        gen = WanMaskGenerator()
        mask_path = gen.generate(
            image_path  = "frog.png",
            moving_zone = [0.0, 0.6, 0.5, 1.0],  # 하단 좌측
            output_dir  = "/tmp/masks/",
        )
    """

    def generate(
        self,
        image_path:  str,
        moving_zone: list[float],
        output_dir:  str | None = None,
    ) -> str:
        """
        마스크 PNG 생성 후 경로 반환.

        Args:
            image_path  : 원본 이미지 경로 (크기 참조용)
            moving_zone : [x1, y1, x2, y2] 상대좌표 (0.0~1.0)
            output_dir  : 저장 디렉토리 (None이면 임시 디렉토리)

        Returns:
            생성된 마스크 PNG 파일 경로
        """
        img = Image.open(image_path)
        W, H = img.size

        x1, y1, x2, y2 = moving_zone
        px1 = int(x1 * W)
        py1 = int(y1 * H)
        px2 = int(x2 * W)
        py2 = int(y2 * H)

        # SetLatentNoiseMask 규칙: 흰색(255)=원본유지, 검정(0)=자유생성
        # → moving zone = 검정(0), fixed zone = 흰색(255)
        mask = Image.new("L", (W, H), 255)  # 전체 흰색 (원본 유지)

        # moving zone = 검정 (WAN이 자유롭게 생성)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([px1, py1, px2, py2], fill=0)

        # 경계 블러 (moving/fixed 전환 완화)
        blur_r = max(2, int(min(W, H) * BLUR_RADIUS_RATIO))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))

        # 저장
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        os.makedirs(output_dir, exist_ok=True)

        stem = Path(image_path).stem
        mask_path = os.path.join(output_dir, f"{stem}_mask.png")
        mask.save(mask_path, "PNG")

        logger.info(
            f"[MaskGen] 마스크 생성: {mask_path} "
            f"(zone=[{px1},{py1},{px2},{py2}] blur={blur_r}px)"
        )
        return mask_path