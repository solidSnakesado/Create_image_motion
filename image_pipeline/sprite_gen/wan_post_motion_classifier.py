"""
image_pipeline/sprite_gen/wan_post_motion_classifier.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

역할:
    [Stage 2] WAN이 생성한 영상을 Gemini Vision으로 분석하여
    키프레임 이동(translate)이 추가로 필요한지 판단.

    Stage 1 (wan_mode_classifier)이 "모션 필요"로 판정 → WAN 생성 완료 후
    이 모듈이 실제 생성된 영상을 보고 최종 판단.

판단 기준:
    생성된 영상에서 대상이 특정 방향으로 이동하는 모션이 있는가?
    → YES: 해당 방향으로 translate 키프레임 추가 (motion + keyframe)
    → NO:  모션만 사용 (motion only)

왜 Stage 2가 필요한가:
    Stage 1에서는 이미지만 보고 "이동이 필요할 것"이라고 예측했지만,
    WAN이 실제로 생성한 모션이 예측과 다를 수 있다.
    - 새 이미지 → Stage 1: "날아갈 것" → WAN 실제: 날개만 접었다 펴기 → 이동 불필요
    - 걷는 사람 → Stage 1: "걸어갈 것" → WAN 실제: 제자리 걷기 → 이동 불필요
    - 정지 개구리 → Stage 1: "제자리 점프" → WAN 실제: 옆으로 도약 → 이동 필요
    실제 영상을 보고 판단해야 정확하다.

설계 원칙:
    - wan_ai_validator.py와 동일한 Gemini Vision 영상 분석 패턴
    - 영상(mp4)을 직접 전달하여 Gemini가 모션 방향을 판단
    - 프롬프트에 특정 오브젝트 카테고리를 예시로 사용하지 않음
    - 2D 평면에서 표현 불가능한 이동(Z축)은 이동 불필요로 판단
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# 출력 모델
# ─────────────────────────────────────────────────────────────

class MotionTravelType(str, Enum):
    """Stage 2 판단 결과."""
    NO_TRAVEL       = "no_travel"        # 순수 제자리 (tail wag, breathing) → 키프레임 불필요
    TRAVEL_LATERAL  = "travel_lateral"   # 좌우 이동 → translateX 키프레임
    TRAVEL_VERTICAL = "travel_vertical"  # 상하 이동 → translateY 키프레임
    TRAVEL_DIAGONAL = "travel_diagonal"  # 대각선 이동 → translateX + Y 키프레임
    AMPLIFY_HOP     = "amplify_hop"      # 제자리 점프/바운스 → translateY 보강
    AMPLIFY_SWAY    = "amplify_sway"     # 좌우 흔들림/몸통 기울기 → translateX 보강
    AMPLIFY_FLOAT   = "amplify_float"    # 부유/떠오름 → translateY 느린 보강


class TravelDirection(str, Enum):
    """이동 방향 (travel이 감지된 경우)."""
    LEFT  = "left"
    RIGHT = "right"
    UP    = "up"
    DOWN  = "down"
    NONE  = "none"


@dataclass
class PostMotionResult:
    """
    Stage 2 분류 결과.

    needs_keyframe      : 키프레임 보강이 필요한가
    travel_type         : 이동/증폭 유형
    travel_direction    : 이동 방향
    confidence          : 판단 확신도 (0.0~1.0)
    reason              : 판단 근거
    suggested_keyframe  : 추천 키프레임 패턴 (hop, bounce, float, nudge_horizontal 등)
    """
    needs_keyframe:     bool
    travel_type:        MotionTravelType
    travel_direction:   TravelDirection
    confidence:         float = 0.0
    reason:             str   = ""
    suggested_keyframe: str   = ""


# ─────────────────────────────────────────────────────────────
# 시스템 프롬프트
# ─────────────────────────────────────────────────────────────

POST_MOTION_PROMPT = """You are analyzing a generated animation video to determine
whether CSS keyframe augmentation would improve the motion on a 2D game screen.

You will receive:
1. (Optional) The ORIGINAL reference image
2. The GENERATED animation video

Your task: Watch the video and classify the motion into one of these categories.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CATEGORIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CATEGORY A — TRAVEL (subject drifts out of frame)
The subject's center of mass moves in a sustained direction and would
leave the frame if the animation continued longer.

  "travel_lateral"  — sustained left or right drift
  "travel_vertical" — sustained up or down drift
  "travel_diagonal" — both horizontal and vertical drift

  → needs_keyframe = true
  → suggested_keyframe: "launch" or "nudge_horizontal" or "nudge_vertical"

CATEGORY B — AMPLIFY (subject moves cyclically but needs position sync)
The subject performs a visible whole-body displacement that returns to
its starting position. The VIDEO already shows the motion, but on a game
screen the sprite would look frozen in place without a matching CSS
translate keyframe to reinforce the movement.

  "amplify_hop"   — the subject jumps UP then lands back down
                     (whole body lifts off the ground, not just limb movement)
  "amplify_sway"  — the subject rocks or leans LEFT-RIGHT noticeably
                     (center of mass shifts horizontally then returns)
  "amplify_float" — the subject slowly drifts UP and DOWN
                     (gentle floating, bobbing, hovering motion)

  → needs_keyframe = true
  → suggested_keyframe: "hop", "bounce", "wobble", or "float"

CATEGORY C — NO_TRAVEL (pure in-place animation, no position change needed)
Only small parts move while the body stays fixed. No whole-body displacement.

  "no_travel" — tail wag, wing flap (without body lift), breathing,
                blinking, head nod, idle sway of appendages only

  → needs_keyframe = false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY DISTINCTION: amplify_hop vs no_travel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Watch whether the ENTIRE BODY lifts off its resting position:
  - Entire body rises visibly → amplify_hop (even if it returns)
  - Only limbs/tail/wings move, body stays → no_travel

A jumping frog, bouncing ball, or hopping character = amplify_hop
A wagging tail, flapping wings without lift = no_travel

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIDENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  0.9~1.0: Very clear motion pattern
  0.6~0.8: Likely but somewhat ambiguous
  0.3~0.5: Subtle, could go either way
  If confidence < 0.5 → force needs_keyframe = false

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond ONLY with JSON. No text outside JSON. No markdown fences.

{
  "needs_keyframe": <true or false>,
  "travel_type": "<no_travel | travel_lateral | travel_vertical | travel_diagonal | amplify_hop | amplify_sway | amplify_float>",
  "travel_direction": "<left | right | up | down | none>",
  "confidence": <float 0.0 to 1.0>,
  "suggested_keyframe": "<hop | bounce | float | wobble | launch | nudge_horizontal | nudge_vertical | (empty string)>",
  "reason": "<1-sentence description of what motion you observed>"
}"""


# ─────────────────────────────────────────────────────────────
# 분류기
# ─────────────────────────────────────────────────────────────

class WanPostMotionClassifier:
    """
    [Stage 2] 생성된 영상을 분석하여 키프레임 이동 필요 여부 판단.

    사용법:
        classifier = WanPostMotionClassifier(api_key="...")
        result = classifier.classify(
            video_path="outputs/wan/bird_attempt01.mp4",
            original_path="outputs/wan/bird_processed.png",
        )
        if result.needs_keyframe:
            # translate 키프레임 추가
            ...
    """

    def __init__(
        self,
        api_key:     str,
        model:       str = "gemini-2.5-flash",
        max_retries: int = 2,
    ) -> None:
        try:
            from google import genai
            self._client      = genai.Client(api_key=api_key)
            self._model       = model
            self._max_retries = max_retries
            self._genai       = genai
        except ImportError as e:
            raise ImportError("pip install google-genai") from e

    def classify(
        self,
        video_path:    str,
        original_path: str,
    ) -> PostMotionResult:
        """
        생성된 영상을 분석. 실패 시 NO_TRAVEL fallback (안전 방향).
        """
        logger.info(f"[Stage2] 모션 분석 시작: {video_path}")

        last_error = None
        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._gemini_analyze(video_path, original_path)
                logger.info(
                    f"[Stage2] 결과: needs_kf={result.needs_keyframe} "
                    f"type={result.travel_type.value} "
                    f"dir={result.travel_direction.value} "
                    f"conf={result.confidence:.2f}"
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[Stage2] 시도 {attempt}/{self._max_retries} 실패: {e}")
                if attempt < self._max_retries:
                    time.sleep(2)

        logger.warning(f"[Stage2] 전부 실패 → NO_TRAVEL fallback: {last_error}")
        return PostMotionResult(
            needs_keyframe   = False,
            travel_type      = MotionTravelType.NO_TRAVEL,
            travel_direction = TravelDirection.NONE,
            confidence       = 0.0,
            reason           = f"Gemini 실패 → no_travel fallback ({last_error})",
        )

    def _gemini_analyze(
        self,
        video_path:    str,
        original_path: str | None,
    ) -> PostMotionResult:
        """Gemini Vision에 영상을 전달하여 분석. 원본 이미지는 선택적."""
        from google.genai import types
        from PIL import Image as PILImage

        # 원본 이미지 로드 (없거나 이미지가 아니면 건너뜀)
        original_img = None
        if original_path and not original_path.endswith((".mp4", ".webm", ".avi")):
            try:
                original_img = PILImage.open(original_path).convert("RGB")
            except Exception:
                logger.warning(f"[Stage2] 원본 이미지 로드 실패, 영상만으로 분석: {original_path}")

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        video_size_mb = len(video_bytes) / (1024 * 1024)

        if video_size_mb < 18:
            video_part = types.Part(
                inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")
            )
        else:
            uploaded = self._client.files.upload(
                file=video_path,
                config=types.UploadFileConfig(mime_type="video/mp4"),
            )
            while uploaded.state.name == "PROCESSING":
                time.sleep(2)
                uploaded = self._client.files.get(name=uploaded.name)
            video_part = types.Part(
                file_data=types.FileData(
                    file_uri=uploaded.uri, mime_type="video/mp4"
                )
            )

        # 원본 이미지가 있으면 포함, 없으면 영상만
        contents = [POST_MOTION_PROMPT]
        if original_img:
            contents.append("Original reference image:")
            contents.append(original_img)
        contents.append("Generated animation video:")
        contents.append(video_part)
        contents.append("Respond with JSON only. No explanation before or after the JSON.")

        response = self._client.models.generate_content(
            model    = self._model,
            contents = contents,
            config   = types.GenerateContentConfig(
                temperature       = 0.1,
                max_output_tokens = 2000,
            ),
        )
        return self._parse_response(response.text)

    @staticmethod
    def _extract_from_text(raw: str) -> dict:
        """JSON 파싱 완전 실패 시 텍스트에서 키워드 추출."""
        raw_lower = raw.lower()
        needs_kf = "true" in raw_lower.split("needs_keyframe")[1][:20] if "needs_keyframe" in raw_lower else False
        
        travel = "no_travel"
        for t in ["amplify_hop", "amplify_sway", "amplify_float",
                   "travel_lateral", "travel_vertical", "travel_diagonal"]:
            if t in raw_lower:
                travel = t
                break

        direction = "none"
        for d in ["left", "right", "up", "down"]:
            if f'"{d}"' in raw_lower:
                direction = d
                break

        suggested = ""
        for s in ["hop", "bounce", "float", "wobble", "launch",
                   "nudge_horizontal", "nudge_vertical"]:
            if f'"{s}"' in raw_lower:
                suggested = s
                break

        return {
            "needs_keyframe": needs_kf,
            "travel_type": travel,
            "travel_direction": direction,
            "confidence": 0.3,
            "suggested_keyframe": suggested,
            "reason": "parsed from incomplete response",
        }

    def _parse_response(self, raw: str) -> PostMotionResult:
        """Gemini 응답 파싱. 불완전한 JSON 복구 시도."""
        clean = re.sub(r"```json|```", "", raw).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # 불완전한 JSON 복구: 잘린 따옴표/중괄호 닫기
            repaired = clean
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            open_braces = repaired.count('{') - repaired.count('}')
            repaired += '}' * max(0, open_braces)
            try:
                data = json.loads(repaired)
                logger.info("[Stage2] JSON 복구 성공")
            except json.JSONDecodeError:
                data = self._extract_from_text(raw)
                logger.info("[Stage2] 키워드 기반 추출 fallback")

        needs_keyframe = bool(data.get("needs_keyframe", False))
        confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))

        # 확신도 낮으면 안전하게 no_travel
        if confidence < 0.5:
            needs_keyframe = False

        # travel_type
        try:
            travel_type = MotionTravelType(data.get("travel_type", "no_travel"))
        except ValueError:
            travel_type = MotionTravelType.NO_TRAVEL

        # no_travel이면 needs_keyframe도 false로 강제
        # (amplify 유형은 needs_keyframe=true 유지)
        if travel_type == MotionTravelType.NO_TRAVEL:
            needs_keyframe = False

        # needs_keyframe=false이면 travel_type도 no_travel로 강제
        if not needs_keyframe:
            travel_type = MotionTravelType.NO_TRAVEL

        # travel_direction
        try:
            direction = TravelDirection(data.get("travel_direction", "none"))
        except ValueError:
            direction = TravelDirection.NONE

        if not needs_keyframe:
            direction = TravelDirection.NONE

        # suggested_keyframe: amplify 유형에 기본 매핑
        suggested = data.get("suggested_keyframe", "")
        if not suggested and needs_keyframe:
            suggested = {
                MotionTravelType.AMPLIFY_HOP:   "hop",
                MotionTravelType.AMPLIFY_SWAY:  "wobble",
                MotionTravelType.AMPLIFY_FLOAT: "float",
                MotionTravelType.TRAVEL_LATERAL:  "nudge_horizontal",
                MotionTravelType.TRAVEL_VERTICAL: "nudge_vertical",
                MotionTravelType.TRAVEL_DIAGONAL: "nudge_horizontal",
            }.get(travel_type, "")

        return PostMotionResult(
            needs_keyframe     = needs_keyframe,
            travel_type        = travel_type,
            travel_direction   = direction,
            confidence         = confidence,
            reason             = data.get("reason", ""),
            suggested_keyframe = suggested,
        )