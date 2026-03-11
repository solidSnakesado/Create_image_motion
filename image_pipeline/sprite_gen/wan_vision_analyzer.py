"""
image_pipeline/sprite_gen/wan_vision_analyzer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    Gemini Vision으로 입력 이미지를 분석하여
    WAN I2V 생성에 필요한 모든 수치와 프롬프트를 결정.

설계 원칙:
    - 분류 없음: AI가 이미지를 보고 모든 것을 자유롭게 결정
    - fallback 없음: Gemini 실패 시 재시도 (최대 3회)
    - 재시도 모두 실패 시 에러 반환
    - MOVING PART ISOLATION: 움직여야 할 최소 신체 부위만 특정
      나머지 신체는 완전 고정 → 검증 통과율 향상
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 출력 모델
# ─────────────────────────────────────────────────────────────

@dataclass
class VisionAnalysisResult:
    object_desc:  str
    action_desc:  str
    moving_parts: str   # 실제 움직이는 신체 부위 설명
    fixed_parts:  str   # 완전 고정되어야 할 신체 부위 설명
    moving_zone:  list  # 마스킹용 moving 영역 bbox [x1,y1,x2,y2] (0.0~1.0 상대좌표)
    frame_rate:   int
    frame_count:  int   # WAN 생성 프레임 수 (모션 유형별 적정 길이)
    min_motion:   float
    max_motion:   float
    max_diff:     float
    positive:     str
    negative:     str
    reason:       str
    pingpong:     bool = True    # True=왕복 루프, False=단방향 루프
    bg_type:      str  = "solid" # "solid"=단색 배경, "scene"=장면 배경
    bg_remove:    bool = True    # 배경 제거 여부
    from_llm:     bool = True


# ─────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """You are an expert in WAN I2V (Image-to-Video) animation generation.
Analyze the image and return ALL parameters needed for a natural, loopable animation.

STEP 1 — IDENTIFY THE SUBJECT
Describe type, color, style, visible body parts.

STEP 2 — CHOOSE THE ACTION
Your goal is to choose the MOST NATURAL motion for this subject — the motion a viewer
would expect to see. Do NOT bias toward safer or simpler motions.
Natural motion produces better results than "easy" motion because WAN responds
to clear, characteristic intent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE A — IDENTIFY THE IDENTITY MOTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Determine what kind of creature or object this is and what motion IS this subject.
Not what motion is safe — what motion makes this subject feel alive.

  SUBJECT TYPE → IDENTITY MOTION (always try this first):
    Bird / flying creature  → wing movement (flap, flutter, or raise/lower)
    Fish / aquatic creature → tail sweep + fin wave (swimming motion)
    Frog / jumping animal   → full body jump cycle
    Quadruped (dog, cat)    → tail wag, ear twitch
    Humanoid / character    → arm wave, torso sway
    Plant / tree            → leaf or branch sway
    Mechanical object       → gear rotation, antenna bob

  PRIORITY RULE: Always attempt the identity motion first.
    Head nods, tail twitches, and micro-motions are LAST RESORT fallbacks,
    used only after the identity motion has failed 3 times.
    Do NOT choose a safer alternative just to avoid risk.

  → Write down the IDENTITY MOTION for this subject before proceeding.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE B — FIND THE EXECUTABLE FORM OF THAT MOTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WAN is a pixel-based model — it cannot fold, unfold, or structurally deform a part.
Find the form of the identity motion that WAN can physically execute.
This is about HOW to express the motion, not WHETHER to attempt it.

Apply these 3 filters to find the right form:

  Q1. IS IT LOOPABLE?
      Can the motion start and end at the same pose?
      ✅ PASS: pendulum swing, tail sweep, wing raise-and-return
      ❌ FAIL: flying off screen, jumping away, mouth staying open

  Q2. IS THE SHAPE PRESERVED DURING MOTION?
      Does the moving part keep its shape, or must it structurally deform?
      ✅ PASS: a wing lifting as a rigid unit, a tail sweeping, fins waving
      ❌ FAIL: a wing that must fold/unfold mid-flap, a mouth opening wide

      If the full form fails Q2, use a PARTIAL form — keep the intent:
        Bird wing full flap (fold/unfold) → ❌ too much deformation
        Bird wing: raises slightly and returns as rigid unit → ✅ same intent, achievable
        Fish full body S-curve wave → ❌ too much deformation
        Fish tail sweeps left-right (rigid) + fins sway gently → ✅ natural swimming feel

  Q3. IS THE MOTION RANGE VISIBLE AND BOUNDED?
      The motion should be clearly visible but not dominate the frame.
      ✅ PASS: motion covers 10–40% of the image area
      ❌ FAIL: invisible micro-motion (<5%) or full-frame takeover (>60%)

Only proceed with a motion that PASSES all 3 filters.
If the identity motion cannot pass in any form, fall back to the next most natural
motion for this subject — NOT the safest or smallest motion available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALWAYS AVOID — regardless of subject type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ❌ Volumetric change only (belly breathing, throat pulsation — no arc through space)
  ❌ Pixel-level changes too small to detect (eye blinking, skin color flush)
  ❌ Whole-body locomotion (subject moves its position across the frame)
  ❌ Body sway or drift with no clear returning motion

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SPECIAL CASE — subjects with powerful jumping legs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  If the subject has large hind legs clearly built for leaping (the legs dominate
  the lower body and are visibly coiled/crouched):
    → A full jump cycle is acceptable: body rises then returns near origin
    → Must be horizontally centered, must complete the loop back near start
    → This is an exception because the whole body moves as a rigid unit (shape preserved)

STEP 3 — ISOLATE MOVING PARTS
Based on YOUR observation of the image, decide:
- MOVING: the single smallest part you identified in STEP 2
- FIXED: everything else — the entire body, head, torso must be completely frozen

The moving part should be visually distinct and small relative to the whole body.

STEP 3.5 — MOVING ZONE BBOX
Estimate the bounding box of the MOVING part in relative coordinates (0.0 to 1.0).
Origin is top-left. x1,y1 = top-left corner, x2,y2 = bottom-right corner.
  - Be GENEROUS: add 20% padding around the actual part so the motion has room.
  - The zone should cover the full arc of motion, not just the resting position.
  - Minimum zone size: 0.15 × 0.15 (to ensure meaningful denoising area)
  - For JUMPING creatures: zone covers the ENTIRE body [0.05, 0.05, 0.95, 0.95]
  Examples:
    Tail at bottom-right → [0.55, 0.60, 0.95, 0.95]
    Head nodding at top  → [0.20, 0.00, 0.80, 0.35]
    Fin on left side     → [0.00, 0.30, 0.35, 0.70]

STEP 4 — FRAME RATE
  Choose based on the natural speed of the chosen motion:
  Fast (small rapid repeating motion): 16–20 fps
  Medium (moderate arc, moderate speed): 12–16 fps
  Slow (large gentle motion): 8–12 fps

STEP 5 — WRITE PROMPTS
Keep prompts SHORT and CLEAR. Do NOT use pixel coordinates or repeat phrases.
⚠️ WRITE BOTH positive AND negative IN CHINESE (简体中文). WAN was trained on Chinese data.

Positive (under 60 words in Chinese):
  Line 1: subject type, color, art style
  Line 2: the moving part and how it moves — describe the motion as a small arc or tilt
  Line 3: which parts stay completely still — explicitly state "身体完全静止，不旋转"
  Line 4: background, loop, quality keywords

  MOTION EXPRESSION GUIDE:
    For small arc/sweep motion: "小幅度来回摆动", "轻轻摆动", "小幅摆动"
    For tilt/nod motion: "轻轻点头", "小幅度倾斜后复位"
    For lift/lower motion: "轻轻抬起后放下", "小幅度上下运动"
    The motion must be described as returning to start: always include "来回" or "后复位"

  FORBIDDEN words (cause whole-body rotation or drift):
    "旋转", "转身", "扭动", "摇摆身体", "全身运动"
    Do NOT use: any phrase that implies the whole body rotates or translates

  For the SPECIAL CASE (full body jump):
    - 身体向上跳起后落回原位附近
    - 保持画面中央，不左右移动
    - 动作完整循环，结束姿势接近开始姿势

Negative (under 40 words in Chinese):
  Focus on the most likely failures for this specific motion.
  Always include: body rotation, body drift, and any volumetric-only changes.
  For small-part motions: also include whole-body movement and the part freezing.
  Always include these two universal failure patterns:
    - 身体消失，角色部位缺失  (body/part disappearing mid-animation)
    - 背景出现灰色区域，背景变色  (grey box or color stain appearing in background)

STEP 6 — MOTION THRESHOLDS
  Set based on the spatial range of the chosen motion:
  Large range (motion covers >20% of image): min=0.05~0.08, max=0.20~0.35
  Medium range (motion covers 10~20% of image): min=0.03~0.06, max=0.12~0.25
  Small range (motion covers <10% of image): min=0.02~0.04, max=0.08~0.15
  max_diff = max_motion × 1.5

STEP 7 — FRAME COUNT
  Decide how many frames WAN should generate (BEFORE pingpong doubling).
  WAN constraint: minimum 9, maximum 81 frames.
  Choose based on the motion type:

  SHORT (17–21 frames): Fast repeating cycles that need only one simple arc.
    Use when: the motion is a small, quick sweep or flick that completes in under 1 second.
    Examples: wing lower-and-return, tail flick, head bob, ear twitch
    Reasoning: fewer frames = less time for WAN to introduce deformation artifacts.

  MEDIUM (25–33 frames): Motions that require a clear arc with visible peak displacement.
    Use when: the motion has a moderate range and needs ~1–1.5 seconds to look natural.
    Examples: tail sweep, fin wave, body sway, leg lift

  LONG (33–49 frames): Full-cycle motions where the complete action must be visible.
    Use when: the motion has multiple distinct phases (launch → peak → land).
    Examples: frog jump (crouch → launch → airborne → land),
              fish full-body S-curve, full wing flap cycle
    Reasoning: cutting these short makes the motion look incomplete or abrupt.

  RULE: When in doubt, choose SHORT over LONG.
  Fewer frames reduce generation artifacts and keep the loop tight.

STEP 8 — LOOP TYPE & BACKGROUND
  Analyze the image to determine:

  A) PINGPONG (loop direction):
     Determine whether the animation should play forward-then-backward (pingpong)
     or forward-only (one-directional loop).

     ━━━ CORE PRINCIPLE ━━━
     Ask yourself: "If this image were a real moment frozen in time,
     would the subject RETURN to the starting state, or CONTINUE forward?"

     Then ask: "Is the subject moving in ONE DIRECTION through space?"
     If the subject, or anything in the scene, is traveling in a single
     direction (forward, away, upward, etc.), pingpong MUST be false.
     The direction of travel does not reverse in reality.

     IMPORTANT: Judge by the POSE, not the background.
     A subject can be walking/running/riding even on a white or solid background.
     Look at the body pose: are the legs mid-stride? Is the body leaning forward?
     Is the subject seen from behind, moving away? These all mean pingpong = false,
     regardless of whether the background is solid white or a detailed scene.

     pingpong = false (continue forward):
       The image implies ongoing movement through space or time.
       The subject's pose suggests travel (walking, running, riding, flying away).
       If you played it backward, it would look unnatural.

     pingpong = true (return to start):
       The image implies a stationary subject doing a repeating motion.
       The subject's pose is static (standing, sitting, perching, hovering).
       Nothing in the scene is traveling in any direction.
       Playing it forward then backward would look natural.

  B) BACKGROUND TYPE:
     bg_type = "solid":
       - Background is white, single color, or simple gradient
       - No environmental elements (no road, sky, trees, buildings)

     bg_type = "scene":
       - Background contains environmental elements (road, sky, forest, room, city)
       - Background is part of the content and should NOT be removed

  C) BACKGROUND REMOVAL:
     bg_remove = true:  when bg_type is "solid" (remove and make transparent)
     bg_remove = false: when bg_type is "scene" (keep background as-is)

OUTPUT FORMAT
Respond ONLY with JSON. No text outside JSON. No markdown fences.

{
  "object_desc": "subject appearance and visible body parts",
  "action_desc": "chosen action",
  "moving_parts": "only the parts that move",
  "fixed_parts": "all parts that stay frozen",
  "moving_zone": [x1, y1, x2, y2],
  "reason": "why this action and part isolation",
  "frame_rate": <integer 8-24>,
  "frame_count": <integer 9-81>,
  "min_motion": <float 0.02-0.25>,
  "max_motion": <float 0.08-0.80>,
  "max_diff": <float max_motion*1.5>,
  "positive": "<중국어 positive 프롬프트, 60단어 이내>",
  "negative": "<중국어 negative 프롬프트, 40단어 이내>",
  "pingpong": <true or false>,
  "bg_type": "<solid or scene>",
  "bg_remove": <true or false>
}"""


# ─────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────

class WanVisionAnalyzer:
    """
    Gemini Vision으로 이미지 분석 → WAN 파라미터 결정.
    분류/fallback 없이 AI가 완전히 자유롭게 판단.
    실패 시 최대 3회 재시도, 모두 실패 시 예외 발생.
    """

    def __init__(
        self,
        api_key:     str,
        model:       str = "gemini-2.5-flash",
        max_retries: int = 3,
    ) -> None:
        try:
            from google import genai
            self._client      = genai.Client(api_key=api_key)
            self._model       = model
            self._max_retries = max_retries
        except ImportError as e:
            raise ImportError("pip install google-genai") from e

    def analyze(self, image_path: str) -> VisionAnalysisResult:
        """
        이미지 분석. 실패 시 최대 3회 재시도.
        모두 실패 시 RuntimeError 발생.
        """
        logger.info(f"[Vision] Gemini 분석 시작: {image_path}")

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._gemini_analyze(image_path)
                logger.info(
                    f"[Vision] 완료 → {result.action_desc} "
                    f"| moving={result.moving_parts[:60]} "
                    f"| fps={result.frame_rate} "
                    f"| motion={result.min_motion}~{result.max_motion}"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[Vision] 시도 {attempt}/{self._max_retries} 실패: {e}"
                )
                if attempt < self._max_retries:
                    time.sleep(2)

        raise RuntimeError(
            f"[Vision] Gemini 분석 {self._max_retries}회 모두 실패: {last_error}"
        )

    def _gemini_analyze(self, image_path: str) -> VisionAnalysisResult:
        """Gemini Vision 호출."""
        from google.genai import types
        from PIL import Image as PILImage

        img = PILImage.open(image_path)

        response = self._client.models.generate_content(
            model    = self._model,
            contents = [VISION_SYSTEM_PROMPT, img],
            config   = types.GenerateContentConfig(
                temperature       = 0.3,
                max_output_tokens = 5000,
            ),
        )
        return self._parse_response(response.text, image_path)

    def _parse_response(self, raw: str, image_path: str) -> VisionAnalysisResult:
        """Gemini 응답 파싱 → VisionAnalysisResult."""
        clean = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(clean)

        frame_rate  = max(8,  min(24,  int(data["frame_rate"])))
        frame_count = max(9,  min(81,  int(data.get("frame_count", 33))))
        min_motion  = max(0.02, min(0.25, float(data["min_motion"])))
        max_motion  = max(min_motion + 0.05, min(1.00, float(data["max_motion"])))
        max_diff    = max(max_motion, min(1.50, float(data["max_diff"])))

        positive = data["positive"].strip()
        negative = data["negative"].strip()

        if len(positive) < 10 or len(negative) < 10:  # 중국어는 글자수 적음
            raise ValueError(
                f"프롬프트가 너무 짧음 "
                f"(positive={len(positive)}, negative={len(negative)})"
            )

        # moving_zone: [x1,y1,x2,y2] 파싱 및 클램핑 (0.0~1.0)
        raw_zone = data.get("moving_zone", [0.0, 0.0, 1.0, 1.0])
        try:
            z = [float(v) for v in raw_zone[:4]]
            x1 = max(0.0, min(1.0, z[0]))
            y1 = max(0.0, min(1.0, z[1]))
            x2 = max(0.0, min(1.0, z[2]))
            y2 = max(0.0, min(1.0, z[3]))
            # 최소 크기 보장 (0.15 × 0.15)
            if x2 - x1 < 0.15:
                cx = (x1 + x2) / 2
                x1, x2 = max(0.0, cx - 0.075), min(1.0, cx + 0.075)
            if y2 - y1 < 0.15:
                cy = (y1 + y2) / 2
                y1, y2 = max(0.0, cy - 0.075), min(1.0, cy + 0.075)
            moving_zone = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
        except Exception:
            moving_zone = [0.0, 0.0, 1.0, 1.0]  # fallback: 전체 영역

        return VisionAnalysisResult(
            object_desc  = data["object_desc"],
            action_desc  = data["action_desc"],
            moving_parts = data.get("moving_parts", ""),
            fixed_parts  = data.get("fixed_parts", ""),
            moving_zone  = moving_zone,
            reason       = data.get("reason", ""),
            frame_rate   = frame_rate,
            frame_count  = frame_count,
            min_motion   = min_motion,
            max_motion   = max_motion,
            max_diff     = max_diff,
            positive     = positive,
            negative     = negative,
            pingpong     = bool(data.get("pingpong", True)),
            bg_type      = data.get("bg_type", "solid"),
            bg_remove    = bool(data.get("bg_remove", True)),
            from_llm     = True,
        )
    def analyze_with_exclusion(
        self,
        image_path:     str,
        exclude_action: str,
    ) -> VisionAnalysisResult:
        """
        특정 액션을 제외하고 재분석.
        unnatural_movement/character_inconsistency 연속 실패 시
        wan_backend이 액션 전환을 위해 호출.
        """
        logger.info(
            f"[Vision] 액션 전환 재분석 (제외: {exclude_action[:60]})"
        )
        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._gemini_analyze_with_exclusion(
                    image_path, exclude_action
                )
                logger.info(
                    f"[Vision] 전환 완료 → {result.action_desc} "
                    f"| moving={result.moving_parts[:60]}"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[Vision] 전환 분석 시도 {attempt} 실패: {e}")
                if attempt < self._max_retries:
                    time.sleep(2)
        raise RuntimeError(
            f"[Vision] 액션 전환 분석 {self._max_retries}회 실패: {last_error}"
        )

    def _gemini_analyze_with_exclusion(
        self,
        image_path:     str,
        exclude_action: str,
    ) -> VisionAnalysisResult:
        """제외 액션을 명시한 프롬프트로 Gemini 호출."""
        from google.genai import types
        from PIL import Image as PILImage

        img = PILImage.open(image_path)

        exclusion_note = (
            f"\n\nIMPORTANT: The following action was already attempted and "
            f"FAILED due to WAN generation issues (unnatural deformation or "
            f"character inconsistency). Do NOT choose this action again:\n"
            f"  EXCLUDED: \"{exclude_action}\"\n"
            f"Choose a DIFFERENT motion that avoids the same moving part."
        )

        response = self._client.models.generate_content(
            model    = self._model,
            contents = [VISION_SYSTEM_PROMPT + exclusion_note, img],
            config   = types.GenerateContentConfig(
                temperature       = 0.5,   # 다양성을 위해 약간 높임
                max_output_tokens = 5000,
            ),
        )
        return self._parse_response(response.text, image_path)