"""
image_pipeline/sprite_gen/wan_mode_classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    [Stage 1] 입력 이미지를 Gemini Vision으로 분석하여
    WAN 모션 생성이 필요한지 판단하는 사전 라우터.

    이 모듈은 "키프레임만 필요한가 vs 모션 생성이 필요한가"만 판단.
    모션 생성 후 키프레임 이동이 추가로 필요한지는
    wan_post_motion_classifier.py (Stage 2)가 영상을 보고 판단.

처리 모드 (Stage 1):
    KEYFRAME_ONLY : 형태 변형 없음 → WAN 불필요.
                    기존 AnimationPipeline (L1→L2A→L2B) 경로.

    MOTION_NEEDED : 형태 변형 필요 → WAN I2V 생성 진행.
                    생성 후 Stage 2에서 키프레임 이동 여부 재판단.

판단 기준:
    PRE-CHECK: 분류 가능한가? + 장면 배경인가?
    Q1: 변형 가능한 부위가 있는가?
    → YES → MOTION_NEEDED
    → NO  → KEYFRAME_ONLY

설계 원칙:
    - 이 단계에서는 이동 방향/키프레임 여부를 판단하지 않음
    - 프롬프트에 특정 오브젝트 카테고리를 예시로 사용하지 않음
    - 판단 기준은 물리적 속성으로만 서술
    - 판단 오류 시 MOTION_NEEDED fallback (WAN이 처리 가능한 안전 방향)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 출력 모델
# ─────────────────────────────────────────────────────────────

class ProcessingMode(str, Enum):
    """Stage 1 판단 결과. 키프레임만 vs 모션 생성 필요."""
    KEYFRAME_ONLY = "keyframe_only"
    MOTION_NEEDED = "motion_needed"


class FacingDirection(str, Enum):
    """이동 방향. KEYFRAME_ONLY일 때 키프레임 방향 결정에 사용."""
    LEFT  = "left"
    RIGHT = "right"
    UP    = "up"
    DOWN  = "down"
    NONE  = "none"


@dataclass
class ModeClassification:
    """
    Stage 1 분류 결과.

    processing_mode   : KEYFRAME_ONLY 또는 MOTION_NEEDED
    facing_direction  : 방향 (KEYFRAME_ONLY에서 키프레임 방향 결정용)
    has_deformable    : Q1 판단 결과
    is_scene          : PRE-CHECK D 결과 (장면 배경 여부)
    subject_desc      : 대상 설명
    reason            : 판단 근거
    suggested_action  : KEYFRAME_ONLY일 때 권장 애니메이션
    """
    processing_mode:  ProcessingMode
    facing_direction: FacingDirection
    has_deformable:   bool
    is_scene:         bool   = False
    subject_desc:     str    = ""
    reason:           str    = ""
    suggested_action: str    = ""


# ─────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────

MODE_CLASSIFIER_PROMPT = """You are an image analysis expert for a game animation pipeline.
Your task is to determine whether a given image needs pixel-level motion generation (WAN)
or can be animated using simple keyframe transforms (translate, rotate, scale).

The pipeline has TWO engines:
  ENGINE A — Keyframe engine (CPU, fast):
    Moves the image as a single rigid unit using translate, rotate, scale, opacity.
    The image pixel content does NOT change — only its position/orientation changes.
    Suitable when the subject's entire body maintains its exact shape during motion.

  ENGINE B — WAN I2V motion engine (GPU, slow, high quality):
    Generates actual pixel-level animation — parts of the image deform frame by frame.
    Suitable when parts of the subject must bend, flex, or change shape during motion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRE-CHECK — before classification
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before answering Q1, check these conditions:

  A) IS THERE A DISCRETE SUBJECT?
     The image must contain a single identifiable object, character, or entity
     with a clear boundary separating it from the background.

     UNCLASSIFIABLE (set processing_mode = "keyframe_only", suggested_action = "pop"):
       - Empty or blank image (no subject)
       - Background/landscape only (no discrete object)
       - Abstract patterns, gradients, or textures with no object boundary
       - Pure text or typography with no pictorial subject

  B) MULTIPLE SUBJECTS?
     If the image contains more than one distinct subject, classify based on
     the LARGEST or most PROMINENT subject (the one occupying the most area).
     Mention this in your reasoning.

  C) FORMLESS / AMORPHOUS SUBJECTS?
     Subjects with no stable shape constantly change form, which is NOT the same
     as having "deformable parts." Deformable parts implies a stable base shape
     that flexes — formless subjects have NO stable base shape at all.
     → set processing_mode = "keyframe_only", suggested_action = "pop" or "wobble".

  D) SCENE IMAGE WITH ENVIRONMENTAL BACKGROUND?
     If the image contains a detailed environmental background (road, sky, trees,
     buildings, room interior, landscape) rather than a plain/solid background:
     → The background itself may need to animate (parallax, flow, ambient motion).
     → This ALWAYS requires the WAN motion engine.
     → Set processing_mode = "motion_needed", is_scene = true, and skip Q1.

     How to distinguish:
       - SOLID background (white, single color, simple gradient, transparent):
         → subject is a standalone sprite → proceed to Q1
       - SCENE background (environmental elements present):
         → subject is embedded in a scene → force "motion_needed"

If the image passes the pre-check, proceed to Q1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Q1. DOES THE SUBJECT HAVE PARTS THAT WOULD CHANGE SHAPE DURING NATURAL MOTION?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Look at the subject in the image and ask:
"If this subject were to perform its most natural motion,
 would any visible part BEND, FLEX, FOLD, or DEFORM?"

The answer depends on WHAT YOU SEE, not what category the subject belongs to.

Answer YES when you observe:
  - Visible joints or articulation points that would bend during motion
  - Appendages attached by a flexible connection (would sway, wave, or curl)
  - Organic or soft-looking structures that would flex under force
  - Any part where the OUTLINE SHAPE would visibly change between frames

Answer NO when you observe:
  - A single solid body with no articulation points
  - All parts are rigidly connected — the whole subject moves as one unit
  - Rotating parts (e.g. wheels, propellers) that spin without changing shape
    → rotation is handled by the keyframe engine, not pixel deformation
  - Surface details (painted eyes, decals, patterns) that are part of the rigid surface
    → these move WITH the body, they don't deform independently

KEY PRINCIPLE:
  The question is NOT "what is this object?"
  The question IS "would any visible part change its outline shape during motion?"
  A subject you've never seen before can still be classified by examining its structure.

→ If YES: mode = "motion_needed"
→ If NO:  mode = "keyframe_only"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACING DIRECTION — for keyframe_only mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When mode is "keyframe_only", determine which direction the subject faces.

Look for these structural cues:
  - The "front" of the subject (where the leading edge, face, or nose points)
  - The direction of body lean or tilt
  - The pointed/tapered end of an elongated subject

  "left"  : front faces toward the left edge of the image
  "right" : front faces toward the right edge
  "up"    : front faces toward the top edge
  "down"  : front faces toward the bottom edge
  "none"  : subject is symmetric with no clear front

For motion_needed mode, set facing_direction based on the subject's orientation
(this may be used later by downstream processing).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUGGESTED ACTION — for keyframe_only mode only
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When mode is "keyframe_only", suggest how the keyframe engine should animate.
Choose based on the subject's shape and facing direction:

  "nudge_horizontal" : translate in the horizontal facing direction
                       → when the subject has a clear left/right front
  "nudge_vertical"   : translate in the vertical facing direction
                       → when the subject has a clear up/down front
  "wobble"           : gentle rocking rotation around center
                       → when the subject has no strong directionality
  "spin"             : Y-axis rotation illusion (scaleX oscillation)
                       → when the subject is round, disc-shaped, or radially symmetric
  "bounce"           : small vertical oscillation
                       → when the subject has a playful or lightweight appearance
  "pop"              : brief scale pulse (1.0 → 1.08 → 1.0)
                       → when the subject is very small or completely static
  "launch"           : one-way accelerating movement in the facing direction, then stop
                       → when the subject has a strong directional thrust posture
                         (pointed tip, streamlined shape, exhaust/trail visible)
  "float"            : slow up-down drifting with slight horizontal sway
                       → when the subject appears lightweight, buoyant, or suspended
                         (round shape, no ground contact, airy/ethereal appearance)
  "parabolic"        : arc trajectory — rises then falls (thrown object path)
                       → when the subject appears to be mid-throw or in free flight
                         (ball-like shape, tilted posture suggesting trajectory)
  "hop"              : vertical jump in place — rises up then returns to original position
                       → when the subject is grounded and appears ready to jump
                         (compact body, legs visible, ground contact, no horizontal bias)

For motion_needed mode, set suggested_action to "".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with JSON. No text outside JSON. No markdown fences.

{
  "is_classifiable": <true or false>,
  "is_scene": <true or false — true if environmental background detected>,
  "subject_desc": "<brief visual description of the subject>",
  "has_deformable_parts": <true or false>,
  "deformable_reasoning": "<describe WHAT structural feature you observed — do not name the object category>",
  "processing_mode": "<keyframe_only | motion_needed>",
  "facing_direction": "<left | right | up | down | none>",
  "suggested_action": "<nudge_horizontal | nudge_vertical | wobble | spin | bounce | pop | launch | float | parabolic | hop | (empty string)>",
  "reason": "<1-sentence summary referencing structural observations, not the object name>"
}"""


# ─────────────────────────────────────────────────────────────
# 분류기
# ─────────────────────────────────────────────────────────────

class WanModeClassifier:
    """
    [Stage 1] 이미지가 WAN 모션 생성이 필요한지 판단.
    키프레임 이동 여부는 판단하지 않음 (Stage 2에서 처리).
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

    def classify(self, image_path: str) -> ModeClassification:
        """
        이미지 분류. 실패 시 최대 3회 재시도.
        모두 실패 시 MOTION_NEEDED fallback.
        """
        logger.info(f"[Stage1] 분류 시작: {image_path}")

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._gemini_classify(image_path)
                logger.info(
                    f"[Stage1] 결과: mode={result.processing_mode.value} "
                    f"facing={result.facing_direction.value} "
                    f"scene={result.is_scene} "
                    f"action={result.suggested_action}"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[Stage1] 시도 {attempt}/{self._max_retries} 실패: {e}")
                if attempt < self._max_retries:
                    time.sleep(2)

        logger.warning(f"[Stage1] 전부 실패 → MOTION_NEEDED fallback: {last_error}")
        return ModeClassification(
            processing_mode  = ProcessingMode.MOTION_NEEDED,
            facing_direction = FacingDirection.NONE,
            has_deformable   = True,
            subject_desc     = "classification failed",
            reason           = f"Gemini 실패 → MOTION_NEEDED fallback ({last_error})",
        )

    def _gemini_classify(self, image_path: str) -> ModeClassification:
        """Gemini Vision 호출."""
        from google.genai import types
        from PIL import Image as PILImage

        img = PILImage.open(image_path)
        response = self._client.models.generate_content(
            model    = self._model,
            contents = [MODE_CLASSIFIER_PROMPT, img],
            config   = types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 1000,
            ),
        )
        return self._parse_response(response.text)

    @staticmethod
    def _extract_from_text(raw: str) -> dict:
        """JSON 파싱 완전 실패 시 텍스트에서 키워드 추출."""
        raw_lower = raw.lower()

        # processing_mode
        mode = "motion_needed"  # 안전 방향 fallback
        if "keyframe_only" in raw_lower:
            mode = "keyframe_only"

        # has_deformable_parts
        has_def = True
        if '"has_deformable_parts"' in raw_lower:
            after = raw_lower.split('"has_deformable_parts"')[1][:20]
            if "false" in after:
                has_def = False

        # is_scene
        is_scene = False
        if '"is_scene"' in raw_lower:
            after = raw_lower.split('"is_scene"')[1][:20]
            if "true" in after:
                is_scene = True

        # facing_direction
        facing = "none"
        for d in ["left", "right", "up", "down"]:
            if f'"{d}"' in raw_lower:
                facing = d
                break

        # suggested_action
        action = ""
        for a in ["nudge_horizontal", "nudge_vertical", "wobble", "spin",
                   "bounce", "pop", "launch", "float", "parabolic", "hop"]:
            if f'"{a}"' in raw_lower:
                action = a
                break

        return {
            "processing_mode": mode,
            "has_deformable_parts": has_def,
            "is_scene": is_scene,
            "facing_direction": facing,
            "suggested_action": action,
            "subject_desc": "parsed from incomplete response",
            "reason": "JSON recovery fallback",
        }

    def _parse_response(self, raw: str) -> ModeClassification:
        """Gemini 응답 파싱. 불완전한 JSON 복구 시도 포함."""
        clean = re.sub(r"```json|```", "", raw).strip()

        # 1차: 정상 파싱
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # 2차: 따옴표/중괄호 수리 후 재파싱
            repaired = clean
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += '}' * max(0, open_braces)
            try:
                data = json.loads(repaired)
                logger.info("[Stage1] JSON 복구 성공 (따옴표/중괄호 수리)")
            except json.JSONDecodeError:
                # 3차: 키워드 기반 추출 fallback
                data = self._extract_from_text(raw)
                logger.info("[Stage1] 키워드 기반 추출 fallback")

        # PRE-CHECK: 분류 불가
        if not data.get("is_classifiable", True):
            return ModeClassification(
                processing_mode  = ProcessingMode.KEYFRAME_ONLY,
                facing_direction = FacingDirection.NONE,
                has_deformable   = False,
                subject_desc     = data.get("subject_desc", "unclassifiable"),
                reason           = data.get("reason", "pre-check failed"),
                suggested_action = "pop",
            )

        is_scene = bool(data.get("is_scene", False))
        has_deformable = bool(data.get("has_deformable_parts", True))

        # PRE-CHECK D: 장면 배경 → 강제 MOTION_NEEDED
        if is_scene:
            mode = ProcessingMode.MOTION_NEEDED
        # Q1: 변형 가능 부위
        elif has_deformable:
            mode = ProcessingMode.MOTION_NEEDED
        else:
            mode = ProcessingMode.KEYFRAME_ONLY

        # mode_str과 실제 판단의 일관성 검증
        mode_str = data.get("processing_mode", "")
        try:
            reported_mode = ProcessingMode(mode_str)
            if reported_mode != mode:
                logger.warning(
                    f"[Stage1] mode 보정: 응답={mode_str} → 판단={mode.value}"
                )
        except ValueError:
            pass

        # facing_direction
        try:
            facing = FacingDirection(data.get("facing_direction", "none"))
        except ValueError:
            facing = FacingDirection.NONE

        # suggested_action
        suggested_action = data.get("suggested_action", "")
        if mode != ProcessingMode.KEYFRAME_ONLY:
            suggested_action = ""
        elif not suggested_action:
            if facing == FacingDirection.NONE:
                suggested_action = "wobble"
            elif facing in (FacingDirection.UP, FacingDirection.DOWN):
                suggested_action = "nudge_vertical"
            else:
                suggested_action = "nudge_horizontal"

        return ModeClassification(
            processing_mode  = mode,
            facing_direction = facing,
            has_deformable   = has_deformable,
            is_scene         = is_scene,
            subject_desc     = data.get("subject_desc", ""),
            reason           = data.get("reason", ""),
            suggested_action = suggested_action,
        )