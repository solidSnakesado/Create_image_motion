"""
image_pipeline/sprite_gen/wan_bg_remover.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    WAN I2V 생성 영상의 배경을 제거하여 투명 PNG 시퀀스 생성.

방식:
    - 각 프레임의 테두리 픽셀 색상을 배경 기준색으로 자동 감지
    - flood fill로 테두리와 연결된 배경 영역 제거
    - 배경색 무관하게 처리 (흰색, 검정, 보라 모두 대응)
    - 캐릭터 내부 색상 보존 (배(belly) 흰색 등)

출력:
    - 투명 PNG 시퀀스 (프레임별)
    - APNG (애니메이션 PNG)
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)


class WanBgRemover:
    """
    WAN 생성 영상 배경 제거기.

    사용법:
        remover = WanBgRemover()
        png_dir = remover.remove_background(
            video_path="outputs/wan/frog_attempt01.mp4",
            output_dir="outputs/transparent/",
        )
    """

    def __init__(self, tolerance: int = 50) -> None:
        """
        Args:
            tolerance: 배경색 판단 허용 오차 (0~255)
                       낮을수록 엄격, 높을수록 넓은 범위 제거
                       기본값 50: WAN 생성 영상의 흰색 배경이
                       회색(~RGB 206-217)으로 생성되는 경우 커버
                       (diff 최대 49 < 50 → 제거됨)
        """
        self.tolerance = tolerance

    def remove_background(
        self,
        video_path:  str,
        output_dir:  str,
        output_apng: bool = True,
        fps:         int  = 16,
    ) -> str:
        """
        영상에서 배경 제거 후 투명 PNG 저장.

        Args:
            video_path  : 입력 MP4 경로
            output_dir  : PNG 저장 디렉토리
            output_apng : APNG 파일도 생성 여부
            fps         : APNG fps

        Returns:
            output_dir 경로
        """
        stem = Path(video_path).stem
        os.makedirs(output_dir, exist_ok=True)

        # 1. 영상 → 프레임 추출
        frames = self._extract_frames(video_path)
        if not frames:
            raise RuntimeError(f"프레임 추출 실패: {video_path}")

        logger.info(f"[BgRemover] {len(frames)}프레임 추출 완료")

        # 2. 배경 기준색 결정 (첫 프레임 테두리 기준)
        bg_color = self._detect_bg_color(frames[0])
        logger.info(
            f"[BgRemover] 배경색 감지: "
            f"R={bg_color[0]:.0f} G={bg_color[1]:.0f} B={bg_color[2]:.0f}"
        )

        # 3. 각 프레임 배경 제거
        transparent_frames = []
        for i, frame in enumerate(frames):
            result = self._remove_bg_frame(frame, bg_color)
            transparent_frames.append(result)

            # PNG 저장
            out_path = os.path.join(output_dir, f"{stem}_frame_{i:04d}.png")
            result.save(out_path, "PNG")

        logger.info(f"[BgRemover] {len(transparent_frames)}프레임 저장 완료: {output_dir}")

        # 4. APNG 생성
        if output_apng:
            apng_path = os.path.join(output_dir, f"{stem}_transparent.apng")
            self._save_apng(transparent_frames, apng_path, fps)
            logger.info(f"[BgRemover] APNG 저장: {apng_path}")

        return output_dir

    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        """영상에서 모든 프레임 추출."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "ffmpeg", "-i", video_path,
                f"{tmpdir}/frame_%04d.png",
                "-y", "-loglevel", "quiet",
            ]
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode != 0:
                return []

            paths = sorted(glob.glob(f"{tmpdir}/frame_*.png"))
            frames = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                frames.append(np.array(img))

        return frames

    def _detect_bg_color(self, frame: np.ndarray) -> np.ndarray:
        """첫 프레임 테두리 픽셀의 평균색 = 배경 기준색."""
        H, W = frame.shape[:2]
        border_size = max(3, H // 20)

        border_pixels = np.concatenate([
            frame[:border_size, :, :].reshape(-1, 3),
            frame[-border_size:, :, :].reshape(-1, 3),
            frame[:, :border_size, :].reshape(-1, 3),
            frame[:, -border_size:, :].reshape(-1, 3),
        ])
        return border_pixels.mean(axis=0)

    def _remove_bg_frame(
        self,
        frame:    np.ndarray,
        bg_color: np.ndarray,
    ) -> Image.Image:
        """
        단일 프레임 배경 제거.
        flood fill: 테두리와 연결된 배경색 픽셀 → 투명 처리.
        """
        H, W = frame.shape[:2]

        # 배경색과 유사한 픽셀 마스크
        is_bg_color = np.all(
            np.abs(frame.astype(int) - bg_color) < self.tolerance,
            axis=2
        )

        # flood fill: 테두리에서 연결된 배경만
        labeled, _ = ndimage.label(is_bg_color)
        border_labels = (
            set(labeled[0, :].tolist()) |
            set(labeled[-1, :].tolist()) |
            set(labeled[:, 0].tolist()) |
            set(labeled[:, -1].tolist())
        )
        border_labels.discard(0)

        bg_mask = np.zeros((H, W), dtype=bool)
        for lbl in border_labels:
            bg_mask |= (labeled == lbl)

        # RGBA로 변환 후 배경 투명 처리
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3] = frame
        rgba[:, :, 3] = 255
        rgba[bg_mask, 3] = 0

        return Image.fromarray(rgba, "RGBA")

    def _save_apng(
        self,
        frames:   list[Image.Image],
        out_path: str,
        fps:      int,
    ) -> None:
        """APNG 저장 (pillow apng 지원).

        disposal=2: 각 프레임 표시 후 투명으로 초기화.
        disposal=0(기본값)은 이전 프레임 잔상이 브라우저에서
        누적되어 회색 박스처럼 보이는 문제가 발생함.
        """
        try:
            duration_ms = int(1000 / fps)
            frames[0].save(
                out_path,
                format="PNG",
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=duration_ms,
                disposal=2,
            )
        except Exception as e:
            logger.warning(f"[BgRemover] APNG 저장 실패: {e}")