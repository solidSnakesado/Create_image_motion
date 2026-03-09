"""
image_pipeline/sprite_gen/wan_validator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

검증 항목:
    1. no_motion          — 정지 영상 (캐릭터 영역 기준)
    2. too_slow           — 액션 대비 너무 느림
    3. too_fast           — 액션 대비 너무 빠름
    4. repeated_motion    — 비정상 반복 (peaks > 8)
    5. frame_escape       — 캐릭터 화면 이탈
    6. no_return_to_origin — 원본 포즈 미복귀
    7. center_drift       — 위치 이동 감지

변경 내역:
    - max_repeat_peaks  : 8 → 12    (pingpong 64프레임 기준 상향)
    - max_return_diff   : 0.40      (pingpong 이음새 허용)
    - center_drift 검증 추가  (drift > 0.12)
    - _get_bg_mask → 첫 프레임 테두리 기반 자동 배경 감지로 통일
    - 동작별 특수 처리 제거: 모든 동작에 동일 기준 적용
      (저난이도 동작 선택 방식으로 변경됨 — vision_analyzer 참조)
    - area_contrib 합산 제거: color_motion만 사용
      (날개 접힘 등 면적 급변 시 char_motion 폭발 방지)
    - too_fast 조건: OR → AND + raw_motion 교차 검증
      (면적 변화로 char_motion이 튀어도 raw_motion 낮으면 통과)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 검증 결과 모델
# ─────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed:        bool
    failed_checks: list[str]        = field(default_factory=list)
    scores:        dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.passed:
            return f"✅ PASS {self.scores}"
        return f"❌ FAIL {self.failed_checks} {self.scores}"


# ─────────────────────────────────────────────────────────────
# 검증기
# ─────────────────────────────────────────────────────────────

class WanValidator:
    """
    WAN I2V 출력 영상 자동 검증기.

    Args:
        max_repeat_peaks       : 허용 최대 동작 반복 횟수 [8]
        max_edge_ratio         : 허용 최대 경계 픽셀 비율 [MD: 0.08]
        check_motion           : 정지 영상 검증 on/off
        check_too_slow         : 너무 느린 동작 검증 on/off
        check_too_fast         : 너무 빠른 동작 검증 on/off
        check_repeat           : 반복 동작 검증 on/off
        check_frame_escape     : 프레임 이탈 검증 on/off
        check_return_to_origin : 원본 복귀 검증 on/off
        max_return_diff        : 첫/마지막 프레임 허용 차이 [0.40]
        check_center_drift     : 위치 이동 검증 on/off [MD: 신규]
        max_center_drift       : 허용 최대 중심 이동량 [MD: 0.12]
    """

    def __init__(
        self,
        max_repeat_peaks:      int   = 12,    # pingpong 64프레임 기준 상향 (8→12)
        max_edge_ratio:        float = 0.08,  # MD: edge_ratio > 0.08
        check_motion:          bool  = True,
        check_too_slow:        bool  = True,
        check_too_fast:        bool  = True,
        check_repeat:          bool  = True,
        check_frame_escape:    bool  = True,
        check_return_to_origin:bool  = True,
        max_return_diff:       float = 0.40,  # pingpong 이음새 허용
        check_center_drift:    bool  = True,  # MD: 신규
        max_center_drift:      float = 0.12,  # MD: drift > 0.12
    ) -> None:
        self.max_repeat_peaks       = max_repeat_peaks
        self.max_edge_ratio         = max_edge_ratio
        self.check_motion           = check_motion
        self.check_too_slow         = check_too_slow
        self.check_too_fast         = check_too_fast
        self.check_repeat           = check_repeat
        self.check_frame_escape     = check_frame_escape
        self.check_return_to_origin = check_return_to_origin
        self.max_return_diff        = max_return_diff
        self.check_center_drift     = check_center_drift
        self.max_center_drift       = max_center_drift

    # ─── 동작 유형 판별 헬퍼 ──────────────────────────────────

    # ─── 메인 검증 ────────────────────────────────────────────

    def validate(self, video_path: str, analysis=None) -> ValidationResult:
        """
        영상 검증.

        Args:
            video_path : 검증할 mp4 파일 경로
            analysis   : VisionAnalysisResult (LLM 결정 motion 기준)
        """
        min_motion = analysis.min_motion if analysis else 0.02
        max_motion = analysis.max_motion if analysis else 0.18
        max_diff   = analysis.max_diff   if analysis else 0.45
        action     = analysis.action_desc if analysis else "unknown"

        logger.info(
            f"[Validator] 검증 시작: {os.path.basename(video_path)} "
            f"(action={action[:60]} motion={min_motion}~{max_motion})"
        )

        frames = self._extract_frames(video_path)
        if not frames:
            return ValidationResult(
                passed=False,
                failed_checks=["frame_extraction_failed"],
            )

        # 첫 프레임 테두리 기반 배경색 자동 감지 (전체 검증에 공유)
        bg_color = self._detect_bg_color(frames[0])

        failed_checks = []
        scores        = {}

        # ── 캐릭터 영역 기준 motion 계산 ─────────────────────

        char_motion = self._calc_character_motion(frames, bg_color, use_area=True)
        raw_motion  = self._calc_motion_score(frames)
        scores["motion"]     = round(char_motion, 4)
        scores["raw_motion"] = round(raw_motion, 4)

        # ── 1. 정지 영상 감지 ─────────────────────────────────
        if self.check_motion:
            threshold = min_motion * 0.5
            if char_motion < threshold:
                failed_checks.append("no_motion")
                logger.debug(
                    f"  [Validator] ❌ no_motion "
                    f"char_motion={char_motion:.4f} < {threshold:.4f}"
                )
            else:
                logger.debug(f"  [Validator] ✅ motion char={char_motion:.4f}")

        # ── 2. 너무 느린 동작 ────────────────────────────────
        if self.check_too_slow:
            if char_motion < min_motion:
                failed_checks.append("too_slow")
                logger.debug(
                    f"  [Validator] ❌ too_slow "
                    f"char_motion={char_motion:.4f} < {min_motion}"
                )
            else:
                logger.debug(f"  [Validator] ✅ not too_slow")

        # ── 3. 너무 빠른 동작 ────────────────────────────────
        if self.check_too_fast:
            max_diff_score = self._calc_max_frame_diff(frames)
            scores["max_diff"] = round(max_diff_score, 4)
            # char_motion은 면적 변화 등으로 부풀려질 수 있으므로
            # raw_motion을 교차 검증에 사용:
            #   char_motion > max_motion  AND  raw_motion > min_motion * 0.5
            # raw_motion이 낮으면 실제로 빠른 게 아니라 면적 변화 오탐으로 판단
            raw_too_fast = raw_motion > (min_motion * 0.5)
            if (char_motion > max_motion and raw_too_fast) or max_diff_score > max_diff:
                failed_checks.append("too_fast")
                logger.debug(
                    f"  [Validator] ❌ too_fast "
                    f"char_motion={char_motion:.4f} > {max_motion} "
                    f"(raw_ok={raw_too_fast}) "
                    f"or max_diff={max_diff_score:.4f} > {max_diff}"
                )
            else:
                logger.debug(f"  [Validator] ✅ not too_fast")

        # ── 4. 반복 동작 감지 ────────────────────────────────
        if self.check_repeat:
            peak_count = self._count_motion_peaks(frames)
            scores["peaks"] = peak_count
            repeat_limit = self.max_repeat_peaks
            if peak_count > repeat_limit:
                failed_checks.append("repeated_motion")
                logger.debug(
                    f"  [Validator] ❌ repeated_motion "
                    f"peaks={peak_count} > {repeat_limit}"
                )
            else:
                logger.debug(f"  [Validator] ✅ peaks={peak_count}")

        # ── bg_drift 선계산 (frame_escape 오판정 억제에 재사용) ─
        # frames는 float32 0~1 스케일 → bg_drift도 0~1 범위
        # 임계값: 30/255 ≈ 0.118 (RGB 30 이상 변화 = 배경 변색으로 판정)
        BG_DRIFT_THRESHOLD = 30 / 255.0
        bg_drift = self._calc_bg_drift(frames)
        scores["bg_drift"] = round(bg_drift * 255)  # 로그용 0~255로 환산

        # ── 5. 프레임 이탈 감지 ──────────────────────────────
        if self.check_frame_escape:
            edge_ratio = self._calc_edge_ratio(frames, bg_color)
            scores["edge_ratio"] = round(edge_ratio, 4)
            if edge_ratio > self.max_edge_ratio:
                # bg_drift>30이면 배경 변색이 원인 → frame_escape는 오판정
                # (배경이 어두워지면 flood-fill이 테두리를 캐릭터로 오인)
                # 실제 캐릭터 이탈 여부는 background_color_change 판정에서 처리
                if bg_drift > BG_DRIFT_THRESHOLD:
                    logger.debug(
                        f"  [Validator] ⚠ edge_ratio={edge_ratio:.4f} 높지만 "
                        f"bg_drift={bg_drift*255:.0f}>30 → 배경 변색 오판정, frame_escape 억제"
                    )
                else:
                    failed_checks.append("frame_escape")
                    logger.debug(
                        f"  [Validator] ❌ frame_escape "
                        f"ratio={edge_ratio:.4f} > {self.max_edge_ratio}"
                    )
            else:
                logger.debug(f"  [Validator] ✅ edge_ratio={edge_ratio:.4f}")

        # ── 6. 원본 복귀 검증 ────────────────────────────────

        if self.check_return_to_origin:
            return_limit = self.max_return_diff
            return_diff  = self._calc_return_diff(frames, bg_color)
            scores["return_diff"] = round(return_diff, 4)
            if return_diff > return_limit:
                failed_checks.append("no_return_to_origin")
                logger.debug(
                    f"  [Validator] ❌ no_return_to_origin "
                    f"diff={return_diff:.4f} > {return_limit}"
                )
            else:
                logger.debug(
                    f"  [Validator] ✅ return_diff={return_diff:.4f} "
                    f"(limit={return_limit})"
                )

        # ── 7. 위치 이동 감지 (center_drift) ─────────────────

        if self.check_center_drift:
            # 수평 drift만 체크 — 수직 이동은 동작 특성상 허용
            drift = self._calc_horizontal_drift(frames, bg_color)
            scores["center_drift"] = round(drift, 4)
            if drift > self.max_center_drift:
                failed_checks.append("center_drift")
                logger.debug(f"  [Validator] ❌ center_drift drift={drift:.4f} > {self.max_center_drift}")
            else:
                logger.debug(f"  [Validator] ✅ center_drift={drift:.4f}")

        # ── 8. 잔상(Ghost) 감지 ─────────────────────────────
        # 첫 프레임에서 배경(흰색)이었던 픽셀 위치에
        # 이후 프레임에서 캐릭터 색과 유사한 픽셀이 나타나면 잔상
        # = 몸통 이동 아티팩트 또는 WAN 픽셀 블렌딩 결과
        ghost_score = self._calc_ghost_score(frames, bg_color)
        scores["ghost_score"] = round(ghost_score, 4)
        GHOST_THRESHOLD = 0.005  # 배경 픽셀 중 0.5% 이상이 잔상이면 FAIL
        if ghost_score > GHOST_THRESHOLD:
            failed_checks.append("ghosting")
            logger.debug(
                f"  [Validator] ❌ ghosting ghost_score={ghost_score:.4f} > {GHOST_THRESHOLD}"
            )
        else:
            logger.debug(f"  [Validator] ✅ ghost_score={ghost_score:.4f}")

        # ── 배경 색상 변화 감지 ──────────────────────────────
        if bg_drift > BG_DRIFT_THRESHOLD:
            # 배경 색상이 변했지만 캐릭터는 정상인지 확인
            # → 캐릭터 픽셀 평균 밝기 변화가 작으면 "배경만 변한 것"
            #   → bg_remover가 배경을 날리므로 최종 APNG에는 영향 없음 → PASS
            # → 캐릭터도 어두워졌다면(전체 필터) → 실제 문제 → FAIL
            char_brightness_drift = self._calc_char_brightness_drift(frames, bg_color)
            scores["char_brightness_drift"] = round(char_brightness_drift * 255)
            if char_brightness_drift > BG_DRIFT_THRESHOLD:
                # 캐릭터까지 어두워짐 = 전체 밝기 변화 → FAIL
                failed_checks.append("background_color_change")
                logger.debug(
                    f"  [Validator] ❌ background_color_change "
                    f"bg_drift={bg_drift*255:.0f} + char_drift={char_brightness_drift*255:.0f} "
                    f"(전체 밝기 변화)"
                )
            else:
                # 배경만 변함 → bg_remover로 해결 가능 → 통과
                logger.debug(
                    f"  [Validator] ⚠ bg_drift={bg_drift*255:.0f} 이지만 "
                    f"char_drift={char_brightness_drift*255:.0f}≤30 → 배경만 변색, 통과"
                )
        else:
            logger.debug(f"  [Validator] ✅ bg_drift={bg_drift*255:.0f}")

        passed = len(failed_checks) == 0
        result = ValidationResult(
            passed        = passed,
            failed_checks = failed_checks,
            scores        = scores,
        )
        logger.info(f"[Validator] 결과: {result}")
        return result

    # ─── 프레임 추출 ─────────────────────────────────────────

    def _extract_frames(self, video_path: str) -> list[np.ndarray]:
        """ffmpeg으로 프레임 추출 → numpy float32 리스트."""
        import subprocess, tempfile, glob

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                "ffmpeg", "-i", video_path,
                "-vf", "scale=240:240",
                f"{tmpdir}/frame_%04d.png",
                "-y", "-loglevel", "quiet",
            ]
            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode != 0:
                logger.error(f"[Validator] ffmpeg 실패: {ret.stderr.decode()}")
                return []

            paths  = sorted(glob.glob(f"{tmpdir}/frame_*.png"))
            frames = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                frames.append(np.array(img, dtype=np.float32) / 255.0)

        return frames

    # ─── 배경색 자동 감지 ────────────────────────────────────

    def _detect_bg_color(self, frame: np.ndarray) -> np.ndarray:
        """
        첫 프레임 테두리 픽셀 평균색 = 배경 기준색.
        흰색 고정 제거 → 보라/검정 등 어떤 배경도 자동 대응.
        """
        H, W, _ = frame.shape
        border_size = max(3, H // 20)
        border_pixels = np.concatenate([
            frame[:border_size, :, :].reshape(-1, 3),
            frame[-border_size:, :, :].reshape(-1, 3),
            frame[:, :border_size, :].reshape(-1, 3),
            frame[:, -border_size:, :].reshape(-1, 3),
        ])
        return border_pixels.mean(axis=0)

    def _get_bg_mask(
        self,
        frame:     np.ndarray,
        bg_color:  np.ndarray,
        tolerance: float = 30,
    ) -> np.ndarray:
        """
        배경 마스크 반환 (True = 배경, False = 캐릭터).
        frames는 float32 0~1 스케일이므로 tolerance도 /255 변환.
        외부에서 0~255 스케일로 넘겨도 내부에서 자동 변환.
        """
        tol = tolerance / 255.0
        return np.all(np.abs(frame.astype(float) - bg_color) < tol, axis=2)

    # ─── 캐릭터 영역 기준 motion 계산 ────────────────────────

    def _calc_character_motion(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
        use_area: bool = False,
    ) -> float:
        """
        배경을 마스킹한 캐릭터 영역에서 픽셀 차이 계산.

        use_area=True:
            색상 차이 + 캐릭터 면적 변화율을 합산.
            날개 펼침/접힘 시 색 변화 없이 면적만 변해도 감지 가능.
            면적 변화율 가중치: × 0.5 (색상 diff 스케일 맞춤)
        """
        if len(frames) < 2:
            return 0.0

        color_diffs = []
        area_changes = []

        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]

            bg1 = self._get_bg_mask(f1, bg_color)
            bg2 = self._get_bg_mask(f2, bg_color)
            char_mask = ~(bg1 & bg2)

            if char_mask.sum() == 0:
                color_diffs.append(0.0)
                if use_area:
                    area_changes.append(0.0)
                continue

            # 색상 기반 motion
            color_diffs.append(float(np.abs(f2.astype(float) - f1.astype(float))[char_mask].mean()))

            # 면적 변화율
            if use_area:
                a1 = (~bg1).sum()
                a2 = (~bg2).sum()
                avg = (a1 + a2) / 2.0
                area_changes.append(abs(a2 - a1) / avg if avg > 0 else 0.0)

        color_motion = float(np.mean(color_diffs)) if color_diffs else 0.0

        if use_area and area_changes:
            # area_contrib은 참고 로그만 남기고 합산하지 않음
            # 날개 접힘처럼 면적이 크게 변하는 케이스에서
            # area_rate가 1.0 이상으로 폭발해 char_motion을 5~20배 부풀리는 문제 방지
            area_rate = float(np.mean(area_changes))
            logger.debug(
                f"  [Validator] area motion: color={color_motion:.4f} "
                f"area_rate={area_rate:.4f} (참고용, 합산 제외)"
            )

        return color_motion

    # ─── 전체 픽셀 motion (raw, 참고용) ──────────────────────

    def _calc_motion_score(self, frames: list[np.ndarray]) -> float:
        """인접 프레임 간 전체 픽셀 차이 평균 (참고용)."""
        if len(frames) < 2:
            return 0.0
        diffs = [
            np.mean(np.abs(frames[i + 1] - frames[i]))
            for i in range(len(frames) - 1)
        ]
        return float(np.mean(diffs))

    def _calc_max_frame_diff(self, frames: list[np.ndarray]) -> float:
        """인접 프레임 간 전체 픽셀 차이 최대값."""
        if len(frames) < 2:
            return 0.0
        diffs = [
            np.mean(np.abs(frames[i + 1] - frames[i]))
            for i in range(len(frames) - 1)
        ]
        return float(np.max(diffs))

    # ─── 반복 동작 감지 ──────────────────────────────────────

    def _count_motion_peaks(self, frames: list[np.ndarray]) -> int:
        """프레임 차이값 피크 수 = 동작 반복 횟수 근사."""
        if len(frames) < 4:
            return 0

        diffs = np.array([
            np.mean(np.abs(frames[i + 1] - frames[i]))
            for i in range(len(frames) - 1)
        ])

        peaks     = 0
        threshold = np.mean(diffs) * 1.5
        for i in range(1, len(diffs) - 1):
            if (diffs[i] > diffs[i - 1] and
                diffs[i] > diffs[i + 1] and
                diffs[i] > threshold):
                peaks += 1

        return peaks

    # ─── 프레임 이탈 감지 ────────────────────────────────────

    def _calc_edge_ratio(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """
        프레임 경계에서 캐릭터 이탈 비율 감지.
        bg_color(자동 감지) 기준으로 테두리의 비배경 픽셀 비율 계산.
        pingpong 이음새(첫 2프레임, 마지막 2프레임)는 제외 — 역방향 전환 시
        순간적으로 픽셀 차이가 크게 잡혀 오판정되는 현상 방지.
        """
        if not frames:
            return 0.0

        # tolerance=0.25: 배경 노란 필터(diff~0.19) 등 배경색 변화를
        # frame_escape로 오판정하지 않도록 여유값 설정
        # (실제 캐릭터 이탈은 테두리에 선명한 캐릭터 픽셀이 나타나므로 diff>0.25)
        tolerance   = 0.25
        edge_ratios = []

        # pingpong 이음새 제외: 앞뒤 2프레임 스킵
        skip = 2
        safe_frames = frames[skip:-skip] if len(frames) > skip * 2 + 2 else frames

        for frame in safe_frames[::2]:
            H, W, _ = frame.shape
            border_size = max(5, H // 10)

            border = np.concatenate([
                frame[:border_size, :, :].reshape(-1, 3),
                frame[-border_size:, :, :].reshape(-1, 3),
                frame[:, :border_size, :].reshape(-1, 3),
                frame[:, -border_size:, :].reshape(-1, 3),
            ])

            non_bg = np.mean(
                np.any(np.abs(border - bg_color) > tolerance, axis=1)
            )
            edge_ratios.append(float(non_bg))

        return float(np.max(edge_ratios))

    # ─── 원본 복귀 확인 ──────────────────────────────────────

    def _calc_return_diff(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """
        첫 프레임과 마지막 프레임의 캐릭터 영역 차이.
        0에 가까울수록 원본 포즈로 잘 복귀함.
        max_return_diff 기준 적용
        """
        if len(frames) < 2:
            return 0.0

        first = frames[0]
        last  = frames[-1]

        bg_first  = self._get_bg_mask(first, bg_color)
        bg_last   = self._get_bg_mask(last,  bg_color)
        char_mask = ~(bg_first & bg_last)

        if char_mask.sum() == 0:
            return 0.0

        return float(np.abs(last.astype(float) - first.astype(float))[char_mask].mean())

    # ─── 위치 이동 감지 (center_drift) ───────────────────────

    def _calc_center_drift(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """
        캐릭터 무게중심의 최대 이동량 계산 (MD 신규 항목).
        수평 이동량 측정.
        기준: max_center_drift = 0.12
        """
        if len(frames) < 2:
            return 0.0

        H, W, _ = frames[0].shape
        centers  = []

        for frame in frames:
            char_mask = ~self._get_bg_mask(frame, bg_color)
            if char_mask.sum() == 0:
                centers.append(None)
                continue
            ys, xs = np.where(char_mask)
            centers.append((xs.mean() / W, ys.mean() / H))

        valid = [c for c in centers if c is not None]
        if len(valid) < 2:
            return 0.0

        ref_cx, ref_cy = valid[0]
        drifts = [
            ((cx - ref_cx) ** 2 + (cy - ref_cy) ** 2) ** 0.5
            for cx, cy in valid[1:]
        ]
        return float(np.max(drifts))
    def _calc_horizontal_drift(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """점프 생물 전용: 수평 drift만 측정 (수직은 제외)."""
        if len(frames) < 2:
            return 0.0

        H, W, _ = frames[0].shape
        cx_list = []

        for frame in frames:
            char_mask = ~self._get_bg_mask(frame, bg_color)
            if char_mask.sum() == 0:
                cx_list.append(None)
                continue
            _, xs = np.where(char_mask)
            cx_list.append(xs.mean() / W)

        valid = [c for c in cx_list if c is not None]
        if len(valid) < 2:
            return 0.0

        ref_cx = valid[0]
        return float(np.max([abs(cx - ref_cx) for cx in valid[1:]]))
    def _calc_bg_drift(self, frames: list[np.ndarray]) -> float:
        """
        배경 색상 변화 감지.
        첫 프레임 테두리 기준 배경색과 이후 프레임 테두리 평균색의
        최대 차이(RGB 절댓값 평균)를 반환.
        attempt07처럼 흰색→회색으로 변하는 경우: diff ≈ 100 → 30 초과로 FAIL.
        """
        if len(frames) < 2:
            return 0.0

        def border_mean(frame: np.ndarray) -> np.ndarray:
            H, W = frame.shape[:2]
            bs = max(3, H // 20)
            pixels = np.concatenate([
                frame[:bs, :].reshape(-1, 3),
                frame[-bs:, :].reshape(-1, 3),
                frame[:, :bs].reshape(-1, 3),
                frame[:, -bs:].reshape(-1, 3),
            ])
            return pixels.mean(axis=0)

        ref_bg = border_mean(frames[0])
        diffs = [
            float(np.abs(border_mean(f).astype(float) - ref_bg).mean())
            for f in frames[1:]
        ]
        return float(np.max(diffs)) if diffs else 0.0

    def _calc_char_brightness_drift(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """
        캐릭터 픽셀의 평균 밝기 변화량 계산.

        배경만 어두워진 경우와 전체(캐릭터+배경)가 어두워진 경우를 구분.
        - 배경만 변함: char_brightness_drift ≤ 30 → bg_remover로 해결 가능
        - 전체 어두워짐: char_brightness_drift > 30 → 실제 품질 문제

        Returns:
            첫 프레임 대비 이후 프레임의 캐릭터 픽셀 밝기 최대 변화량 (0~255)
        """
        if len(frames) < 2:
            return 0.0

        # 첫 프레임 기준 캐릭터 마스크 (배경 아닌 픽셀)
        ref_char_mask = ~self._get_bg_mask(frames[0], bg_color)
        if ref_char_mask.sum() == 0:
            return 0.0

        ref_brightness = frames[0][ref_char_mask].mean()
        drifts = []
        for frame in frames[1::3]:  # 3프레임 간격으로 샘플링
            char_mask = ~self._get_bg_mask(frame, bg_color)
            if char_mask.sum() < 100:
                continue
            brightness = frame[char_mask].mean()
            drifts.append(abs(float(brightness) - float(ref_brightness)))

        return float(np.max(drifts)) if drifts else 0.0
    def _calc_ghost_score(
        self,
        frames:   list[np.ndarray],
        bg_color: np.ndarray,
    ) -> float:
        """
        잔상(Ghosting) 감지 스코어.

        첫 프레임에서 배경(흰색)이었던 픽셀 위치에
        이후 프레임에서 캐릭터 색과 유사한 픽셀이 나타나는 비율을 반환.

        - 정상 케이스: ~0.0005 (0.05%)
        - 잔상 케이스: ~0.01   (1.0%)
        - 임계값: 0.005 (0.5%)

        Returns:
            float: 0.0~1.0 범위 (배경 픽셀 중 잔상 비율)
        """
        if len(frames) < 2:
            return 0.0

        ref = frames[0]
        # 첫 프레임 배경 마스크 (흰색 계열)
        bg_mask = self._get_bg_mask(ref, bg_color)
        bg_count = bg_mask.sum()
        if bg_count < 100:
            return 0.0

        # 캐릭터 평균색 (첫 프레임 기준)
        char_mask = ~bg_mask
        if char_mask.sum() == 0:
            return 0.0
        char_color = ref[char_mask].mean(axis=0)  # float32

        max_ghost = 0.0
        for frame in frames[1::2]:  # 2프레임 간격 샘플링
            # 배경이었던 위치의 현재 픽셀
            current_at_bg = frame[bg_mask]  # (N, 3) float32
            # 캐릭터 색까지의 거리 (RGB 평균 절댓값)
            color_dist = np.abs(current_at_bg - char_color).mean(axis=1)
            # 캐릭터 색에 80/255 이내 = 잔상으로 판정
            ghost_ratio = float(np.mean(color_dist < (80 / 255.0)))
            if ghost_ratio > max_ghost:
                max_ghost = ghost_ratio

        return max_ghost