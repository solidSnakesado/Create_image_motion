"""
image_pipeline/sprite_gen/wan_keyframe_generator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    KEYFRAME_ONLY 판정된 이미지에 대해 CSS 키프레임 애니메이션 생성.
    WAN 모션 생성 없이 키프레임만으로 움직임을 표현.

    Stage 1 (wan_mode_classifier)의 출력:
      - suggested_action: nudge_horizontal, nudge_vertical, wobble, spin, bounce, pop
      - facing_direction: left, right, up, down, none

    이 모듈은 suggested_action을 기존 AnimationType에 매핑하고,
    물리 수식 기반으로 키프레임을 생성한다.

출력:
    AnimationConfig 호환 JSON — 프론트엔드 엔진이 Web Animation API로 실행.
    기존 debug_viewer.html의 playAnim()과 동일한 포맷.

매핑:
    nudge_horizontal → NUDGE (translateX)
    nudge_vertical   → NUDGE (translateY)
    wobble           → WOBBLE (rotate oscillation)
    spin             → SPIN (scaleX oscillation)
    bounce           → JUMP (translateY bounce)
    pop              → POP (scale pulse)
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 키프레임 데이터 (기존 models.py Keyframe과 동일 구조)
# ─────────────────────────────────────────────────────────────

@dataclass
class KFKeyframe:
    """단일 키프레임. debug_viewer.html 엔진 호환."""
    t:           float
    translateX:  float = 0.0
    translateY:  float = 0.0
    rotate:      float = 0.0
    scaleX:      float = 1.0
    scaleY:      float = 1.0
    opacity:     float = 1.0
    glow_color:  Optional[str]   = None
    glow_radius: Optional[float] = None

    def to_dict(self) -> dict:
        d = {
            "t": round(self.t, 3),
            "translateX": round(self.translateX, 2),
            "translateY": round(self.translateY, 2),
            "rotate": round(self.rotate, 2),
            "scaleX": round(self.scaleX, 3),
            "scaleY": round(self.scaleY, 3),
            "opacity": round(self.opacity, 3),
        }
        if self.glow_color is not None:
            d["glow_color"] = self.glow_color
        if self.glow_radius is not None:
            d["glow_radius"] = round(self.glow_radius, 1)
        return d


@dataclass
class KeyframeAnimConfig:
    """KEYFRAME_ONLY 모드의 애니메이션 출력."""
    animation_type:   str
    keyframes:        list[KFKeyframe]
    duration_ms:      int
    easing:           str
    transform_origin: str
    loop:             bool = True
    suggested_action: str  = ""
    facing_direction: str  = ""

    def to_dict(self) -> dict:
        return {
            "animation_type":  self.animation_type,
            "keyframes":       [kf.to_dict() for kf in self.keyframes],
            "duration_ms":     self.duration_ms,
            "easing":          self.easing,
            "transform_origin": self.transform_origin,
            "loop":            self.loop,
        }

    def to_json(self, path: str) -> str:
        """JSON 파일로 저장."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"[KeyframeGen] 저장: {path}")
        return path


# ─────────────────────────────────────────────────────────────
# 물리 수식 기반 키프레임 생성
# ─────────────────────────────────────────────────────────────

def _damped_sin(t: float, amplitude: float, omega: float, lam: float) -> float:
    """감쇠 정현파: A·sin(ωt)·e^(-λt)"""
    return amplitude * math.sin(omega * t) * math.exp(-lam * t)

def _damped_cos(t: float, amplitude: float, omega: float, lam: float) -> float:
    """감쇠 여현파: A·cos(ωt)·e^(-λt)"""
    return amplitude * math.cos(omega * t) * math.exp(-lam * t)


class WanKeyframeGenerator:
    """
    KEYFRAME_ONLY 이미지용 키프레임 생성기.

    사용법:
        gen = WanKeyframeGenerator()
        config = gen.generate(
            suggested_action="nudge_horizontal",
            facing_direction="left",
        )
        config.to_json("outputs/racecar_keyframe.json")
    """

    def generate(
        self,
        suggested_action: str,
        facing_direction: str = "none",
        duration_ms:      int | None = None,
        loop:             bool = True,
    ) -> KeyframeAnimConfig:
        """
        suggested_action에 따라 키프레임 생성.

        Args:
            suggested_action: nudge_horizontal/nudge_vertical/wobble/spin/bounce/pop
            facing_direction: left/right/up/down/none
            duration_ms:      재생 시간 (None이면 기본값)
            loop:             루프 재생 여부
        """
        method = {
            "nudge_horizontal": self._gen_nudge_h,
            "nudge_vertical":   self._gen_nudge_v,
            "wobble":           self._gen_wobble,
            "spin":             self._gen_spin,
            "bounce":           self._gen_bounce,
            "pop":              self._gen_pop,
            "launch":           self._gen_launch,
            "float":            self._gen_float,
            "parabolic":        self._gen_parabolic,
            "hop":              self._gen_hop,
        }.get(suggested_action, self._gen_wobble)

        config = method(facing_direction, duration_ms)
        # loop: 메서드가 명시적으로 설정한 값 우선, 인자는 None일 때만 적용
        if loop is not None:
            # launch, parabolic은 내부에서 loop=False로 설정하므로 유지
            if config.animation_type not in ("launch", "parabolic"):
                config.loop = loop
        config.suggested_action = suggested_action
        config.facing_direction = facing_direction

        logger.info(
            f"[KeyframeGen] {suggested_action} facing={facing_direction} "
            f"→ {config.animation_type} {config.duration_ms}ms "
            f"{len(config.keyframes)} keyframes"
        )
        return config

    # ── NUDGE HORIZONTAL ─────────────────────────────────────

    def _gen_nudge_h(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """좌우 이동 감쇠 진동."""
        duration = dur or 1200
        direction = -1 if facing == "left" else 1
        distance = 20  # px

        # 감쇠 정현파로 8프레임 샘플링
        omega = 2 * math.pi * 2  # 2 cycle
        lam = 1.5
        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)
            time_sec = t * (duration / 1000)
            dx = direction * _damped_sin(time_sec, distance, omega, lam)
            kfs.append(KFKeyframe(t=t, translateX=dx))

        # 마지막 프레임 초기 위치로
        kfs[-1].translateX = 0.0

        return KeyframeAnimConfig(
            animation_type="nudge",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-out",
            transform_origin="center center",
        )

    # ── NUDGE VERTICAL ───────────────────────────────────────

    def _gen_nudge_v(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """상하 이동 감쇠 진동."""
        duration = dur or 1000
        direction = -1 if facing == "up" else 1
        distance = 18

        omega = 2 * math.pi * 2
        lam = 1.8
        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)
            time_sec = t * (duration / 1000)
            dy = direction * _damped_sin(time_sec, distance, omega, lam)
            kfs.append(KFKeyframe(t=t, translateY=dy))

        kfs[-1].translateY = 0.0

        return KeyframeAnimConfig(
            animation_type="nudge",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-out",
            transform_origin="center center",
        )

    # ── WOBBLE ────────────────────────────────────────────────

    def _gen_wobble(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """좌우 감쇠 회전."""
        duration = dur or 1500
        amplitude = 12  # degrees
        omega = 2 * math.pi * 2.5
        lam = 1.2
        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)
            time_sec = t * (duration / 1000)
            angle = _damped_sin(time_sec, amplitude, omega, lam)
            # 약간의 scaleY 변화로 탄성감
            sy = 1.0 + abs(angle) / amplitude * 0.04
            kfs.append(KFKeyframe(t=t, rotate=angle, scaleY=sy))

        kfs[-1].rotate = 0.0
        kfs[-1].scaleY = 1.0

        return KeyframeAnimConfig(
            animation_type="wobble",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-in-out",
            transform_origin="center center",
        )

    # ── SPIN ──────────────────────────────────────────────────

    def _gen_spin(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """Y축 회전 (scaleX 진동)."""
        duration = dur or 1000
        kfs = [
            KFKeyframe(t=0.00, scaleX=1.00, scaleY=1.00),
            KFKeyframe(t=0.25, scaleX=0.02, scaleY=1.08),
            KFKeyframe(t=0.50, scaleX=1.00, scaleY=1.04),
            KFKeyframe(t=0.75, scaleX=0.02, scaleY=1.08),
            KFKeyframe(t=1.00, scaleX=1.00, scaleY=1.00),
        ]

        return KeyframeAnimConfig(
            animation_type="spin",
            keyframes=kfs,
            duration_ms=duration,
            easing="linear",
            transform_origin="center center",
        )

    # ── BOUNCE ────────────────────────────────────────────────

    def _gen_bounce(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """소규모 바운스 (jump 축소판)."""
        duration = dur or 800
        height = 15  # px
        kfs = [
            KFKeyframe(t=0.00, translateY=0,       scaleX=1.00, scaleY=1.00),
            KFKeyframe(t=0.15, translateY=0,       scaleX=1.10, scaleY=0.90),  # squash
            KFKeyframe(t=0.40, translateY=-height,  scaleX=0.92, scaleY=1.12),  # peak
            KFKeyframe(t=0.60, translateY=-height*0.3, scaleX=1.02, scaleY=0.98),
            KFKeyframe(t=0.75, translateY=0,       scaleX=1.06, scaleY=0.94),  # land squash
            KFKeyframe(t=0.90, translateY=-height*0.1, scaleX=0.98, scaleY=1.02),
            KFKeyframe(t=1.00, translateY=0,       scaleX=1.00, scaleY=1.00),
        ]

        return KeyframeAnimConfig(
            animation_type="jump",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-out",
            transform_origin="center bottom",
        )

    # ── POP ───────────────────────────────────────────────────

    def _gen_pop(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """미세 스케일 펄스."""
        duration = dur or 500
        kfs = [
            KFKeyframe(t=0.00, scaleX=1.00, scaleY=1.00),
            KFKeyframe(t=0.30, scaleX=1.08, scaleY=1.08),
            KFKeyframe(t=0.60, scaleX=1.03, scaleY=1.03),
            KFKeyframe(t=1.00, scaleX=1.00, scaleY=1.00),
        ]

        return KeyframeAnimConfig(
            animation_type="pop",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-out",
            transform_origin="center center",
        )

    # ── LAUNCH (한 방향 이동 후 멈춤) ─────────────────────────

    def _gen_launch(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """
        한 방향으로 가속 이동 후 도착점에서 멈춤.
        로켓, 화살, 투사체 등.
        facing에 따라 이동 방향 결정.
        loop=False (1회 재생 후 도착점에서 정지).
        """
        duration = dur or 2000
        distance = 300  # px

        # facing에 따른 이동 벡터
        dx, dy = 0, 0
        if facing == "up":
            dy = -distance
        elif facing == "down":
            dy = distance
        elif facing == "left":
            dx = -distance
        elif facing == "right":
            dx = distance
        else:
            dy = -distance  # 기본: 위로

        # ease-out 곡선을 키프레임으로 표현 (초반 빠르게, 후반 감속)
        kfs = [
            KFKeyframe(t=0.00, translateX=0,           translateY=0),
            KFKeyframe(t=0.05, translateX=dx*0.02,     translateY=dy*0.02,  scaleX=1.02, scaleY=0.98),
            KFKeyframe(t=0.15, translateX=dx*0.15,     translateY=dy*0.15,  scaleX=1.0,  scaleY=1.0),
            KFKeyframe(t=0.35, translateX=dx*0.50,     translateY=dy*0.50),
            KFKeyframe(t=0.55, translateX=dx*0.75,     translateY=dy*0.75),
            KFKeyframe(t=0.75, translateX=dx*0.90,     translateY=dy*0.90),
            KFKeyframe(t=0.90, translateX=dx*0.97,     translateY=dy*0.97),
            KFKeyframe(t=1.00, translateX=dx,          translateY=dy),
        ]

        return KeyframeAnimConfig(
            animation_type="launch",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-out",
            transform_origin="center center",
            loop=False,
        )

    # ── FLOAT (부유 — 느린 상하 진동) ────────────────────────

    def _gen_float(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """
        위아래로 느리게 떠다니는 부유 모션.
        풍선, 구름, 유령, 수중 물체 등.
        부드러운 사인파 + 약간의 좌우 흔들림 + 미세 회전.
        """
        duration = dur or 3000
        float_height = 20    # px (상하 진폭)
        sway_x = 5           # px (좌우 미세 흔들림)
        tilt = 3             # deg (미세 회전)

        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)
            phase = t * 2 * math.pi  # 1 full cycle

            # 상하: -sin (위로 갔다가 내려옴)
            dy = -float_height * math.sin(phase)
            # 좌우: cos (위에서 오른쪽, 아래에서 왼쪽)
            dx = sway_x * math.cos(phase)
            # 회전: sin (상승 시 한쪽으로, 하강 시 반대)
            rot = tilt * math.sin(phase)
            # 스케일: 위에 있을 때 약간 확대 (원근감)
            sy = 1.0 + 0.02 * math.sin(phase)

            kfs.append(KFKeyframe(
                t=t,
                translateX=round(dx, 2),
                translateY=round(dy, 2),
                rotate=round(rot, 2),
                scaleY=round(sy, 3),
            ))

        # 루프 이음새: 첫 프레임과 마지막 프레임 일치
        kfs[-1].translateX = kfs[0].translateX
        kfs[-1].translateY = kfs[0].translateY
        kfs[-1].rotate = kfs[0].rotate
        kfs[-1].scaleY = kfs[0].scaleY

        return KeyframeAnimConfig(
            animation_type="float",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-in-out",
            transform_origin="center center",
            loop=True,
        )

    # ── PARABOLIC (포물선 — 던지기/날아가기) ──────────────────

    def _gen_parabolic(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """
        포물선 궤적 이동.
        던져진 공, 날아가는 물체 등.
        x(t) = Vx·t (등속 수평 이동)
        y(t) = -Vy·t + 0.5·g·t² (상승 후 하강)
        facing에 따라 수평 방향 결정.
        loop=False.
        """
        duration = dur or 1500
        range_x = 200   # px (수평 이동 거리)
        peak_y = 120    # px (최고점 높이)

        direction = -1 if facing == "left" else 1

        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)

            # 수평: 등속 이동
            dx = direction * range_x * t

            # 수직: 포물선 y = -4h·t(t-1) → 0에서 시작, 0.5에서 peak, 1에서 0
            dy = -4 * peak_y * t * (1 - t)

            # 회전: 이동 방향으로 약간 기울기
            rot = direction * 15 * (1 - 2 * t)  # 상승 시 기울고 하강 시 반대

            # 스케일: 정점에서 약간 축소 (원근감)
            s = 1.0 - 0.08 * math.sin(math.pi * t)

            kfs.append(KFKeyframe(
                t=round(t, 3),
                translateX=round(dx, 2),
                translateY=round(dy, 2),
                rotate=round(rot, 2),
                scaleX=round(s, 3),
                scaleY=round(s, 3),
            ))

        return KeyframeAnimConfig(
            animation_type="parabolic",
            keyframes=kfs,
            duration_ms=duration,
            easing="linear",
            transform_origin="center center",
            loop=False,
        )

    # ── HOP (제자리 점프 — 올라갔다 내려오기) ─────────────────

    def _gen_hop(self, facing: str, dur: int | None) -> KeyframeAnimConfig:
        """
        제자리에서 위로 올라갔다가 원래 위치로 내려오는 점프.
        bounce와 비슷하지만 squash/stretch 없이 순수 수직 이동.
        높이가 더 크고 체공 시간이 있음.
        loop=True (반복 점프).
        """
        duration = dur or 1200
        height = 60  # px

        # 포물선 상승-체공-하강: y = -4h·t(t-1)
        steps = 8
        kfs = []
        for i in range(steps):
            t = i / (steps - 1)
            # 포물선: 0에서 시작, 0.5에서 정점, 1에서 0
            dy = -4 * height * t * (1 - t)

            kfs.append(KFKeyframe(
                t=round(t, 3),
                translateY=round(dy, 2),
            ))

        # 마지막 프레임 원점 (루프 이음새)
        kfs[-1].translateY = 0.0

        return KeyframeAnimConfig(
            animation_type="hop",
            keyframes=kfs,
            duration_ms=duration,
            easing="ease-in-out",
            transform_origin="center bottom",
            loop=True,
        )