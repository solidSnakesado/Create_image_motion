"""
image_pipeline/sprite_gen/wan_ai_validator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    Gemini Vision이 생성된 영상을 직접 보고 검증 + 수정안 결정.
    사람이 영상을 보고 판단하는 것처럼 AI가 개입.

검증 항목:
    1. 속도         — 너무 빠르거나 느린지
    2. 원본 복귀    — 마지막 프레임이 원본 포즈로 돌아왔는지
    3. 화면 이탈    — 캐릭터가 화면 밖으로 나갔는지
    4. 자연스러움   — 움직임이 자연스러운지
    5. 캐릭터 일관성 — 원본 이미지와 캐릭터가 동일한지

실패 시:
    Gemini가 직접 수정안 결정
    - frame_rate 조정
    - positive/negative 프롬프트 수정
    - scale 조정
    고정값 없음 — 매번 영상을 보고 판단
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# AI 검증 결과 모델
# ─────────────────────────────────────────────────────────────

@dataclass
class AIValidationResult:
    """
    Gemini Vision 검증 결과.

    passed       : 최종 통과 여부
    issues       : 발견된 문제 목록
    reason       : 판단 근거 (Gemini 설명)
    frame_rate   : 수정된 fps (None이면 변경 없음)
    scale        : 수정된 scale (None이면 변경 없음)
    positive     : 수정된 포지티브 프롬프트 (None이면 변경 없음)
    negative     : 수정된 네거티브 프롬프트 (None이면 변경 없음)
    """
    passed:     bool
    issues:     list[str]       = field(default_factory=list)
    reason:     str             = ""
    frame_rate: int   | None    = None
    scale:      float | None    = None
    positive:   str   | None    = None
    negative:   str   | None    = None

    def __str__(self) -> str:
        if self.passed:
            return f"✅ AI PASS"
        return f"❌ AI FAIL {self.issues} | {self.reason[:80]}"


# ─────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────

AI_VALIDATOR_SYSTEM_PROMPT = """You are an expert animation quality reviewer for WAN I2V (Image-to-Video) generated animations.

You will be given:
1. The ORIGINAL image (reference)
2. Five frames from the generated animation: FIRST(0%), QUARTER(25%), MIDDLE(50%), THREE-QUARTER(75%), LAST(100%)

IMPORTANT — ANIMATION STRUCTURE:
This animation uses pingpong playback: the video plays forward then reverses.
  - FIRST frame = start pose
  - QUARTER frame = peak of motion (maximum displacement from start)
  - MIDDLE frame = on the way back (should look similar to QUARTER but returning)
  - THREE-QUARTER frame = returning toward start
  - LAST frame = should be close to FIRST frame (loop complete)

Therefore: do NOT judge motion by comparing FIRST vs MIDDLE — they may look similar.
Instead: judge motion by comparing FIRST vs QUARTER (25%) — this shows the actual range of movement.
If QUARTER frame shows clear displacement from FIRST frame, the animation IS moving.

Your job is to review the animation quality AS IF YOU WERE A HUMAN watching the video.

═══════════════════════════════════════════════════════
REVIEW CRITERIA
═══════════════════════════════════════════════════════

[1] SPEED
  - Does the motion look too fast or too slow for the action?
  - A frog jump should be snappy but not blurry-fast
  - A bird wing flap should be rapid and continuous
  - A fish swim should be gentle and fluid
  - IMPORTANT: If the numerical validator already PASSED this animation, speed is within acceptable range.
    Only flag speed_too_fast if motion is visually blurry or strobing due to excessive speed.
    Only flag speed_too_slow if the character is barely moving (nearly static).

  CRITICAL — no_motion rule:
    If the numerical validator PASSED this animation, do NOT flag no_motion.
    The numerical validator confirmed that pixel-level motion exists.
    Even if the INTENDED body part (e.g. head) is not visibly moving,
    if the overall character shows any natural subtle motion (body sway, breathing-like tremor),
    this is a PASS. Sprite animations with gentle ambient motion are acceptable.
    → no_motion should only appear when the character is completely frozen with zero movement.

[2] RETURN TO ORIGIN
  - Compare LAST frame to FIRST frame
  - The character MUST return to the same pose it started in
  - If the last frame shows a different pose, this is a failure

[3] FRAME ESCAPE
  - Is any part of the character cut off or outside the frame?
  - Check all four edges of the image
  - Even partial cutoff (limb going out of frame) is a failure

[4] NATURALNESS OF MOVEMENT
  - Does the motion look physically natural?
  - Are limbs moving in unnatural directions?
  - Is the body distorting, stretching, or morphing?
  - Does the animation flow smoothly between frames?
  - Would a human animator be satisfied with this movement?
  - IMPORTANT: WAN I2V is AI-generated, so minor imperfections are acceptable.
    A slightly imperfect but recognizable wing flap / tail swing / leg kick = PASS.
    Do NOT fail for subtle style variation in the moving part — focus on whether the MOTION INTENT is achieved.

  SPECIFIC FAILURE PATTERNS — apply to ANY motion type (flap, swim, jump, twitch, sway, etc.)
  These are universal animation quality failures, not specific to any body part or creature.

  ❌ GHOSTING (ZERO TOLERANCE — check this FIRST):
     A semi-transparent residue of the previous frame bleeds into the current frame.
     This is a HARD FAIL with NO exceptions — do NOT excuse it as motion blur.

     TWO FORMS — both are FAIL:

     Form 1 — OUTLINE GHOST: A pale grey or white trailing residue clinging to the
     OUTLINE of the moving part, as if the part moved but left its silhouette behind.
     Ask: "Is there a ghostly border or smear around the outline of the moving part?"

     Form 2 — BODY DRIFT GHOST: The main body (which should be FROZEN) drifts
     slightly, leaving a semi-transparent copy of its previous position visible
     alongside the current position. The character appears to have two overlapping
     versions of itself — the new position and a faded ghost of the old position.
     Ask: "Does the body appear in two slightly offset positions simultaneously,
           with one being faint/translucent?"

     KEY DISTINCTION from acceptable motion blur:
     → Natural motion blur: the MOVING PART streaks in the direction of travel,
       background stays clean white. PASS.
     → Ghosting: translucent character-colored pixels appear in the WHITE BACKGROUND
       area. The body or outline appears doubled. FAIL.
     Even subtle ghosting visible on close inspection = FAIL.

  ❌ FLICKERING: The moving part changes shape abruptly between frames —
     instead of tracing a smooth continuous arc, it snaps to a different shape suddenly,
     creating a flickering or strobing visual effect.
     Ask: "Does the moving part smoothly travel through space, or does it jump/snap?"

  ❌ TEXTURE RECONSTRUCTION: The surface detail of the moving part
     (spots, stripes, texture, markings) is visibly regenerated each frame
     rather than moving as part of a unified structure.
     The part appears to 'repaint itself' rather than physically move.
     Ask: "Does the surface detail shift/disappear/reappear independently of the motion?"

  ❌ PART INDEPENDENCE: Parts that should be completely frozen
     (any body part NOT intended to move) undergo VISIBLE shape change, deformation,
     or texture reconstruction throughout the animation.
     Ask: "Are parts that should be still actually CHANGING SHAPE or DEFORMING?"

     IMPORTANT distinction:
     → FAIL: Wing goes blurry/smeared, body shape visibly distorts, texture pattern changes
     → FAIL: A fixed limb clearly bends, stretches, or morphs into a different shape
     → PASS: Fixed parts show subtle micro-vibration or 1-2px trembling (natural WAN behavior)
     → PASS: "Subtle body jiggling" or "slight wing tremor" — these are acceptable ambient motion
     The threshold is VISIBLE DEFORMATION, not pixel-level trembling.

  ❌ INCOHERENT MOTION RHYTHM: The motion does not follow a smooth physical arc.
     Instead of accelerating and decelerating naturally (like a pendulum or sine wave),
     the moving part trembles, vibrates, or lurches randomly.
     Ask: "Does the motion feel like a clean sweep, or does it feel like shaking/trembling?"

  PASS examples (acceptable minor imperfections):
  ✅ The MOVING PART's outline is slightly soft at the PEAK OF MOTION (natural motion blur)
     — this applies ONLY to the intended moving part, NOT to the frozen body/torso
  ✅ Minor color variation in the moving part during motion
  ✅ Slight positional offset at the loop seam
  ✅ 2D CARTOON SPECIFIC: Subtle pixel jiggling or a slight breathing/swaying effect
     on the body is completely NORMAL in WAN-generated 2D animations.
     Do NOT flag this as 'unnatural_movement' or 'character_inconsistency'
     unless the structural shape actually breaks, blurs, or deforms visibly.

  ⚠ IMPORTANT DISTINCTION — motion blur vs ghosting:
     Natural motion blur: the MOVING PART appears slightly soft/streaked in the direction
     of travel. The background behind it stays clean white.
     Ghosting: a semi-transparent COPY of the body/part appears at a different position.
     The background shows discoloration or translucent character-colored pixels.
     → If the frozen BODY appears soft, translucent, or doubled → always GHOSTING FAIL
     → "Slightly soft" is only acceptable for the actively moving part at peak motion

[5] CHARACTER CONSISTENCY
  - Compare each frame to the ORIGINAL image
  - Does the character look the same (color, style, proportions)?
  - Has the face or expression changed unexpectedly?
  - IMPORTANT: The MOVING PART will look different across frames — this is expected and NOT a failure.
    Only fail if NON-MOVING parts change: body shape, head shape, color scheme, art style.
  - PASS: The intended moving part changes position/shape as part of its motion
  - FAIL: Any body part that should be frozen changes shape, color, or proportion unexpectedly
  - FAIL: The character's overall art style, color palette, or proportions shift across frames
  - FAIL: Any fixed part continuously deforms or ripples frame-by-frame (unintended animation)
  - FAIL: Any major body part (wing, body, head, tail) DISAPPEARS or becomes invisible mid-animation.
    Ask: "Is a large part of the character's body missing in any frame compared to the first frame?"
    A body part that was clearly visible in frame 1 but is gone or mostly gone in frame 2 or 3 = FAIL.
    This includes: wing dissolving into background, body fading out, limbs vanishing.

[6] BACKGROUND COLOR
  Background changes fall into TWO categories — treat them differently:

  CATEGORY A — Background-only discoloration (character is fine):
    The background corners/edges changed color (grey, dark, etc.) BUT the character itself
    remains sharp, correctly colored, and fully intact.
    → This is ACCEPTABLE. The background will be removed in post-processing.
    → Do NOT mark as background_color_change. Set passed=true if all other checks pass.
    → Example: white background turned grey, but the bird looks perfect → PASS

  CATEGORY B — Whole-frame brightness shift (character also affected):
    The entire image darkened or shifted color — the character looks dimmer, washed out,
    or tinted compared to frame 1. Both background AND character are affected.
    → This is a FAIL (background_color_change).
    → Example: everything turned dark brown/black including the character → FAIL

  ALSO CHECK: a grey/cream rectangular patch or halo appearing AROUND the character
    (not just corners) while the character itself looks fine.
    → If the halo covers the character or blends into it → FAIL (character_inconsistency)
    → If the halo is only in the background area → ACCEPTABLE (Category A)

═══════════════════════════════════════════════════════
IF ISSUES ARE FOUND — PROVIDE SPECIFIC ADJUSTMENTS
═══════════════════════════════════════════════════════
You must provide concrete fixes, NOT vague suggestions.

For NO MOTION or TOO SLOW (character barely moves):
  - STEP A: Look at the current positive prompt and identify which body part is currently targeted.
  - STEP B: Judge whether that body part can produce VISIBLE motion in WAN.
    Apply this filter — the target part FAILS if any of these are true:
      ❌ It only swells/pulsates without sweeping through space (belly, throat)
      ❌ Its pixel change is too small to detect (eyelid, skin color)
      ❌ It must structurally deform to perform the motion (folding, bending at joint)
    A part PASSES if: it moves through space while keeping its shape (sweep, tilt, lift, rotate)
  - STEP C: If the current target FAILS the filter → SWITCH to a different part.
    Look at the character in the video and find a part that:
      ✅ Is clearly visible and distinct from the body
      ✅ Can sweep/tilt/rotate while keeping its shape
      ✅ Is small relative to the whole character (roughly 10–25% of the image)
    Do NOT suggest specific body parts by name — choose based on what is visible in this specific image.
  - STEP D: Rewrite positive_addition for the new target (in Chinese).
    Pattern: "[该部位] 轻轻运动，保持形状不变，小幅度来回摆动"
    The part must move as a rigid unit — no folding, no deformation.
  - STEP E: Increase frame_rate by 2 (e.g., 12→14, 14→16, 16→18).
  - STEP F: negative_addition is MANDATORY for no_motion/too_slow.
    Add: any part currently moving that should be frozen, plus "身体晃动，身体摇摆"
    e.g. if wrong part moves: add "[that part]移动，[that part]运动"
  - STEP G: NEVER copy motion phrases from positive into negative — this cancels the motion.
    Negative describes unwanted outcomes only, never the intended motion itself.
  - NEVER amplify an already-failed target — switch instead.

For speed issues (too fast):
  - Decrease frame_rate (e.g., 10, 12, 14)
  - Add to negative_addition: "动作过快，快速移动，急速运动" to reinforce the slow-down

For naturalness/character issues:
  - NEGATIVE IS PRIMARY: first identify the exact visual problem and block it with negative
    e.g. body distortion → "身体变形，身体拉伸，身体扭曲"
    e.g. unnatural arc → "[该部位]运动不自然，[该部位]轨迹突变，[该部位]抖动"
    e.g. character style change → "风格改变，线条模糊，颜色失真，卡通风格丢失"
    e.g. part morphing → "[该部位]变形，[该部位]拉伸，形状改变"
    Replace [该部位] with the actual moving part observed in the video.
  - POSITIVE is secondary: only add if the specific motion needs reinforcing
    e.g. "[该部位]平滑运动，保持形状" (only if the moving part's motion needs reinforcing)
  - RULE: negative_addition must ALWAYS be provided for naturalness/character issues
    positive_addition may be null if negative alone is sufficient

For frame escape:
  - Suggest scale reduction (e.g., 0.55 instead of 0.65)

For background color change (background turns black, purple, grey, yellow):
  - This is a critical WAN generation failure
  - Do NOT change frame_rate — set frame_rate to null
  - Do NOT change scale — set scale to null
  - Only add background prompts, do not touch motion prompts
  - Add to positive (in Chinese): "纯白色背景，整个动画过程中背景始终保持白色"
  - Add to negative (in Chinese): "背景变色，背景变暗，背景变黑，背景变灰，背景变黄，背景变紫，背景颜色偏移，非白色背景"

For return-to-origin:
  - Add to positive: "动作完整循环，结束时回到接近开始的姿势，保持画面中央"
  - Add to negative: "动作不完整，停在最高点，动作中途结束，水平方向漂移"
  - NOTE: for motions where the whole body moves as a rigid unit (e.g. a full body jump/leap),
    exact pixel return is NOT required — returning close to origin is acceptable.
    Only fail if the subject drifts far from center or freezes at peak without returning.

═══════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════
Respond ONLY with JSON. No text outside JSON. No markdown fences.

CRITICAL: Keep the "reason" field under 20 words. Do NOT use newlines, quotes, or
special characters inside the reason string — this will break JSON parsing.

{
  "passed": <true|false>,
  "issues": ["speed_too_fast"|"speed_too_slow"|"no_motion"|"no_return_to_origin"|"frame_escape"|"unnatural_movement"|"character_inconsistency"|"background_color_change"],
  "reason": "one short sentence, under 20 words, no quotes or newlines",
  "adjustments": {
    "frame_rate": <integer or null>,
    "scale": <float or null>,
    "positive_addition": "<specific phrases to ADD to positive prompt, or null>",
    "negative_addition": "<specific phrases to ADD to negative prompt, or null>"
  }
}"""


# ─────────────────────────────────────────────────────────────
# AI 검증기
# ─────────────────────────────────────────────────────────────

class WanAIValidator:
    """
    Gemini Vision으로 생성된 영상을 직접 보고 검증.

    Args:
        api_key : Google AI Studio API 키
        model   : Gemini 모델명
    """

    def __init__(
        self,
        api_key: str,
        model:   str = "gemini-2.5-flash",
    ) -> None:
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._model  = model
            self._genai  = genai
        except ImportError as e:
            raise ImportError("pip install google-genai") from e

    def validate(
        self,
        video_path:    str,
        original_path: str,
        current_fps:   int,
        current_scale: float,
        positive:      str,
        negative:      str,
    ) -> AIValidationResult:
        """
        Gemini Vision으로 영상 검증.

        Args:
            video_path    : 검증할 영상 경로
            original_path : 원본 이미지 경로
            current_fps   : 현재 fps
            current_scale : 현재 scale
            positive      : 현재 포지티브 프롬프트
            negative      : 현재 네거티브 프롬프트

        Returns:
            AIValidationResult
        """
        logger.info(f"[AIValidator] Gemini 검증 시작: {Path(video_path).name}")

        # Gemini 호출 (최대 2회 재시도)
        last_error = None
        for attempt in range(1, 3):
            try:
                result = self._gemini_validate(
                    video_path    = video_path,
                    original_path = original_path,
                    current_fps   = current_fps,
                    current_scale = current_scale,
                    positive      = positive,
                    negative      = negative,
                )
                logger.info(f"[AIValidator] 결과: {result}")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[AIValidator] 시도 {attempt}/2 실패: {e}")
                if attempt < 2:
                    import time
                    time.sleep(2)

        # 모두 실패 → 실패 처리 (통과 처리 금지)
        logger.warning(f"[AIValidator] Gemini 2회 실패 → 실패 처리: {last_error}")
        return AIValidationResult(
            passed=False,
            issues=["ai_validator_error"],
            reason=f"AI 검증 실패 ({last_error})",
            frame_rate=None,
            scale=None,
            positive=None,
            negative=None,
        )

    def _gemini_validate(
        self,
        video_path:    str,
        original_path: str,
        current_fps:   int,
        current_scale: float,
        positive:      str,
        negative:      str,
    ) -> AIValidationResult:
        """Gemini Vision 호출 및 결과 파싱."""
        from google.genai import types
        from PIL import Image as PILImage

        # 원본 이미지 로드
        original_img = PILImage.open(original_path).convert("RGB")

        # 영상 파일 직접 전달 (20MB 미만 → inline)
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_size_mb = len(video_bytes) / (1024 * 1024)

        # 컨텍스트 프롬프트
        context = (
            f"Current parameters: fps={current_fps}, scale={current_scale}\n"
            f"Current positive prompt: {positive}\n"
            f"Current negative prompt: {negative}\n\n"
            f"Images/video provided in order:\n"
            f"1. ORIGINAL reference image\n"
            f"2. The full animation video\n\n"
            f"ANIMATION STRUCTURE: This video uses pingpong playback "
            f"(plays forward then reverses back to start). "
            f"Watch the full video to judge the range of motion, "
            f"whether the character returns to its starting pose, "
            f"and whether the loop is seamless.\n\n"
            f"Please review the animation quality carefully."
        )

        if video_size_mb < 18:
            # 인라인 전달
            video_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
            )
        else:
            # File API 업로드
            import time
            uploaded = self._client.files.upload(
                file=video_path,
                config=types.UploadFileConfig(mime_type="video/mp4"),
            )
            # 처리 완료 대기
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self._client.files.get(name=uploaded.name)
            video_part = types.Part(
                file_data=types.FileData(
                    file_uri=uploaded.uri, mime_type="video/mp4"
                )
            )

        contents = [
            AI_VALIDATOR_SYSTEM_PROMPT,
            context,
            original_img,
            video_part,
        ]

        response = self._client.models.generate_content(
            model    = self._model,
            contents = contents,
            config   = types.GenerateContentConfig(
                temperature       = 0.2,
                max_output_tokens = 4000,
            ),
        )

        return self._parse_response(response.text)

    def _parse_response(self, raw: str) -> AIValidationResult:
        """Gemini 응답 파싱 → AIValidationResult."""
        clean = re.sub(r"```json|```", "", raw).strip()

        # 1차 시도: 정상 파싱
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # 2차 시도: reason 필드 내 줄바꿈/따옴표로 인한 파싱 실패 복구
            # reason은 로그용이므로 제거 후 재파싱
            recovered = re.sub(
                r'"reason"\s*:\s*"(?:[^"\\]|\\.)*"',
                '"reason": ""',
                clean,
                flags=re.DOTALL,
            )
            try:
                data = json.loads(recovered)
                logger.debug("[AIValidator] JSON 복구 파싱 성공 (reason 필드 제거)")
            except json.JSONDecodeError:
                # 3차 시도: adjustments 블록만 추출
                # passed/issues 두 필드만 있어도 동작 가능
                try:
                    passed_m  = re.search(r'"passed"\s*:\s*(true|false)', clean)
                    issues_m  = re.search(r'"issues"\s*:\s*(\[[^\]]*\])', clean)
                    passed_v  = (passed_m.group(1) == "true") if passed_m else True
                    issues_v  = json.loads(issues_m.group(1)) if issues_m else []
                    data = {"passed": passed_v, "issues": issues_v, "reason": "", "adjustments": {}}
                    logger.debug("[AIValidator] JSON 최소 필드 복구 성공")
                except Exception as e2:
                    raise json.JSONDecodeError(str(e2), clean, 0)

        passed      = bool(data.get("passed", True))
        issues      = data.get("issues", [])
        reason      = data.get("reason", "")
        adjustments = data.get("adjustments", {})

        frame_rate = adjustments.get("frame_rate")
        scale      = adjustments.get("scale")
        pos_add    = adjustments.get("positive_addition")
        neg_add    = adjustments.get("negative_addition")

        # 타입 보정
        if frame_rate is not None:
            frame_rate = max(10, min(24, int(frame_rate)))
        if scale is not None:
            scale = max(0.40, min(0.80, float(scale)))

        return AIValidationResult(
            passed     = passed,
            issues     = issues,
            reason     = reason,
            frame_rate = frame_rate,
            scale      = scale,
            positive   = pos_add,
            negative   = neg_add,
        )