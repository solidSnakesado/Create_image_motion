# WAN I2V 파이프라인 개발 히스토리
> 1주차: 2026-03-06 ~ 2026-03-08 | 2주차: 2026-03-09 ~ 2026-03-10 | 3~4주차: 2026-03-11 ~ 2026-03-16
> 목적: 검증 개발 과정 + 프롬프트 개선 과정 + 기능 수정 기록 + 환경 구성 기록

---

## 스크립트 구성

**wan_backend.py** (메인 오케스트레이터)
전체 흐름을 제어합니다.
> 이미지 전처리(480 캔버스 배치),
> ComfyUI API 통신(업로드/큐/다운로드), 생성 루프(MAX_RETRIES=7),
> 실패 시 보완 조치(AI 조정/seed 교체/액션 전환),
> 성공 시 후처리 합성 + 배경 제거, VRAM/RAM 초기화, 검증 통계 기록까지 모든 것을 조율.

**wan_vision_analyzer.py** (이미지 분석)
생성 전에 Gemini Vision으로 원본 이미지를 분석.
> "이 이미지에서 어떤 부위를 어떻게 움직일 것인가"를 결정.
> 액션, 프롬프트(중국어), fps, 프레임 수,
> moving zone, pingpong, bg_type, bg_remove를 모두 AI가 판단하여 반환.

**wan_validator.py** (수치 검증)
생성된 영상을 수치 8항목으로 검증.
> motion(움직임량), ghosting(잔상), no_motion(무움직임),
> too_slow(느린 움직임), frame_escape(프레임 이탈), background_color_change(배경 변색),
> repeated_motion(반복 모션), no_return_to_origin(원점 미복귀)을 프레임 간 픽셀 차이로 계산.

**wan_ai_validator.py** (AI 검증)
수치 검증 통과 후 Gemini Vision으로 영상 품질을 추가 검증.
> 사람 눈으로 봤을 때 자연스러운지 판단.
> unnatural_movement, character_inconsistency, speed_too_slow 등을 감지,
> 실패 시 프롬프트 조정 힌트(fps 변경, negative 추가)를 반환.

**wan_mask_generator.py** (마스크 생성)
Vision Analyzer가 결정한 moving zone을 흑백 마스크 PNG로 생성.
> 검정(0)=움직이는 영역, 흰색(255)=고정 영역. 후처리 합성에서 고정 부위를 원본으로 덮어쓸 때 사용.
> 경계에 Gaussian blur를 적용하여 자연스러운 전환을 생성.

**wan_bg_remover.py** (배경 제거)
성공한 영상에서 배경을 제거하여 투명 PNG 시퀀스 + APNG + WebM을 생성.
> 첫 프레임 테두리 색상으로 배경색을 감지하고, flood fill로 테두리와 연결된 배경만 투명 처리.
> 캐릭터 내부 색상(흰색 배 등)은 보존.

**wan_mode_classifier.py** (Stage 1 모드 분류) — 3~4주차 추가
이미지가 WAN 모션 생성이 필요한지 사전 판단.
> KEYFRAME_ONLY(강체) → WAN 건너뜀, MOTION_NEEDED(변형 가능) → WAN 생성 진행.
> 10가지 suggested_action 중 하나를 결정 (물리적 속성 기반, 오브젝트 하드코딩 없음).

**wan_post_motion_classifier.py** (Stage 2 모션 후 분류) — 3~4주차 추가
WAN 생성 영상을 Gemini Vision으로 분석하여 키프레임 보강 필요 여부 판단.
> TRAVEL(이동) / AMPLIFY(점프·부유 증폭) / NO_TRAVEL(제자리) 3가지 카테고리.
> suggested_keyframe(hop, bounce, float 등)을 함께 반환.

**wan_keyframe_generator.py** (키프레임 생성) — 3~4주차 추가
KEYFRAME_ONLY 판정 또는 Stage 2 amplify 감지 시 CSS 키프레임 자동 생성.
> 10가지 모션 패턴 (물리 수식 기반: 감쇠 정현파, 포물선, 사인파).

**wan_lottie_converter.py** (Lottie 변환) — 3~4주차 추가
투명 PNG 시퀀스를 Lottie JSON 애니메이션으로 변환.
> 래스터 내장(base64), 4가지 프리셋 (original/web/web_hd/mobile).

**wan_server.py** (REST API 서버) — 3~4주차 추가
대시보드 프론트엔드용 Flask REST API. 기존 파이프라인 코드 무수정.

**wan_dashboard.html** (대시보드 프론트엔드) — 3~4주차 추가
5단계 워크플로우 시각적 테스트 도구. 키프레임 스테이지 뷰 + Lottie 동기화.

### 실행 순서
```
이미지 입력
  → wan_mode_classifier (Stage 1: KEYFRAME_ONLY vs MOTION_NEEDED)
  │
  ├─ KEYFRAME_ONLY → wan_keyframe_generator → JSON 저장 → 완료
  │
  └─ MOTION_NEEDED:
       → wan_vision_analyzer (분석)
       → wan_mask_generator (마스크)
       → wan_backend (ComfyUI로 생성)
       → wan_validator (수치 검증)
       → wan_ai_validator (AI 검증)
       → wan_backend (후처리 합성 with 마스크)
       → wan_bg_remover (배경 제거 → APNG + WebM)
       → wan_lottie_converter (Lottie JSON 변환)
       → wan_post_motion_classifier (Stage 2: TRAVEL / AMPLIFY / NO_TRAVEL)
       → wan_keyframe_generator (amplify 시 키프레임 보강)
```

---

## 목차
1. [1주차 개발 개요](#1-개발-개요)
2. [파이프라인 전체 구조 (현재)](#2-파이프라인-전체-구조)
3. [개발 히스토리 (시간순)](#3-개발-히스토리-시간순)
   - 3-1. [기반 구축 — ComfyUI 연동 + 워크플로우](#3-1-기반-구축)
   - 3-2. [최초 테스트 — 물고기 + 화질 문제](#3-2-최초-테스트-물고기)
   - 3-3. [Validator 구현 + 버그 수정](#3-3-validator-구현)
   - 3-4. [프롬프트 전략 전환 — 영어→중국어](#3-4-영어→중국어)
   - 3-5. [개구리 테스트 — 점프 사이클 문제](#3-5-개구리-테스트)
   - 3-6. [pingpong 도입 + AI Validator 비디오 직접 피드](#3-6-pingpong)
   - 3-7. [물고기 APNG 완성 + frame_escape 오판정 수정](#3-7-물고기-apng)
   - 3-8. [새(bird) 테스트 — BG 문제 + 날개 flap 10회 실패](#3-8-새-테스트)
   - 3-9. [마스킹 파이프라인 도입 + 후처리 합성 전환](#3-9-마스킹)
   - 3-10. [합성 순서 버그 + 액션 전환 로직 추가](#3-10-합성-순서)
   - 3-11. [수치 검증 대규모 오탐 수정](#3-11-수치-검증-오탐)
   - 3-12. [Ghosting 감지 강화 + 프롬프트 누적 제거](#3-12-ghosting)
   - 3-13. [암전 방지 + SOFT_ISSUES 수정 + 테스트 캐릭터 생성](#3-13-암전)
   - 3-14. [successful_results 루프 수정 + 이미지 분리](#3-14-successful-results)
4. [검증 시스템 발전 과정](#4-검증-시스템-발전-과정)
5. [프롬프트 개선 과정](#5-프롬프트-개선-과정)
6. [핵심 설계 결정 배경 (Why)](#6-핵심-설계-결정-배경)
7. [최종 파일 현황 + 설정값](#7-최종-파일-현황)
8. [2주차 개발 히스토리](#8-2주차-개발-히스토리)
   - 8-1. [Git 레포 셋업 + 코드 업로드](#8-1-git-레포-셋업)
   - 8-2. [WSL 새 환경 셋업 + 검증](#8-2-wsl-새-환경-셋업)
   - 8-3. [RTX 50XX PyTorch 호환성 해결](#8-3-pytorch-호환성)
   - 8-4. [ComfyUI 버전 고정 (VRAM 행 방지)](#8-4-comfyui-버전-고정)
   - 8-5. [VRAM 누수 문제 확인 + 초기화 기능 추가](#8-5-vram-초기화)
   - 8-6. [검증 실패 통계 + 보완 조치 추적 기능](#8-6-검증-통계)
   - 8-7. [pingpong/bg_type/bg_remove AI 동적 판단](#8-7-pingpong-동적-판단)
9. [3~4주차 개발 히스토리](#9-3~4주차-개발-히스토리)
   - 9-1. [Stage 1 모드 분류기](#9-1-stage-1-모드-분류기)
   - 9-2. [Stage 2 모션 후 분류기](#9-2-stage-2-모션-후-분류기)
   - 9-3. [키프레임 생성기 (10종)](#9-3-키프레임-생성기)
   - 9-4. [Lottie 변환기](#9-4-lottie-변환기)
   - 9-5. [대시보드 프론트엔드](#9-5-대시보드-프론트엔드)
   - 9-6. [출력 폴더 분리](#9-6-출력-폴더-분리)
   - 9-7. [Lottie + CSS 키프레임 동기화](#9-7-lottie-css-키프레임-동기화)
10. [PENDING](#10-pending)

---

## 1. 개발 개요

### 두 파이프라인 공존

**기존 파이프라인** (`pipeline.py`, `layer1_decision.py`, `layer2a_physics.py`, `layer2b_ai_keyframe.py`):
- 스프라이트 PNG → 규칙/물리 기반 + AI Keyframe → CSS/Web Animation
- LLM이 Keyframe 좌표를 직접 설계, CPU 배포 가능

**WAN I2V 파이프라인**:
- 스프라이트 PNG → Gemini Vision 분석 → ComfyUI WAN 모델 → 수치+AI 이중 검증 → 배경 제거 → APNG
- GPU 필수, Wan2.1-i2v-14b 모델 사용

### 핵심 설계 원칙 (완성본 기준)
1. **AI 주도** — 고정 규칙 없이 Gemini가 동작/수치/프롬프트 전부 결정
2. **IN-PLACE 모션** — 위치 이동 없이 제자리 동작만 (center_drift 검증으로 감지)
3. **아이코닉한 동작** — 미세 동작(호흡, 눈 깜빡임) 금지, WAN이 표현 가능한 큰 동작만
4. **분류 코드 없음** — 동물/탈것 분류 기반 분기 전체 제거 (Gemini가 자유 판단)

---

## 2. 파이프라인 전체 구조

### 현재 설정값
| 항목 | 값 | 비고 |
|---|---|---|
| 해상도 | 480×480 | |
| steps | 20 | 원래 30 → 33% 시간 단축 |
| pingpong | **AI 동적 결정** | 포즈 기반 판단: 정지=True, 이동=False |
| frames | AI 동적 결정 | 17~81 범위, 권장 17~49 |
| MAX_RETRIES | **5** | 이미지당 최대 생성 시도 횟수 |
| CONSECUTIVE_FAIL | **2** | 연속 실패 시 액션 전환 임계값 |
| WAN 모델 | wan2.1-i2v-14b-480p-Q3_K_S.gguf | |
| Gemini 모델 | gemini-2.5-flash | 분석 + 검증 모두 사용 |

### 실행 흐름 (`wan_backend.py`)
```
generate(image_path)
  │
  ├─ Step 0: preprocess_image_simple()
  │    이미지를 480×480 흰색 캔버스에 배치 (scale=0.65, headroom_top=0.18)
  │    _white_anchor(): 배경 순백색 강제 + 경계 선명화
  │    검정 배경 이미지도 자동 흰 배경으로 변환됨
  │
  ├─ Step 1: WanVisionAnalyzer.analyze()
  │    Gemini Vision으로 이미지 분석 → 액션·프롬프트·수치 전부 결정
  │    + pingpong (포즈 기반 이동 감지), bg_type (solid/scene), bg_remove 판단
  │
  ├─ Step 1.5: BG 프롬프트 분기
  │    bg_type=solid → 흰색 배경 보호 문구 (BG_POSITIVE/BG_NEGATIVE)
  │    bg_type=scene → 배경 유지 문구 ("背景保持不变")
  │
  ├─ Step 2: ComfyUIClient.upload_image()
  │
  ├─ Step 2.5: WanMaskGenerator.generate()
  │    moving_zone bbox → 흑백 마스크 PNG 생성
  │
  └─ Step 3: for attempt in range(1, MAX_RETRIES + 1):
       ├─ VRAM 측정 (nvidia-smi)
       ├─ _generate_one() → ComfyUI 생성 (pingpong=analysis.pingpong) → mp4 다운로드
       ├─ WanValidator.validate() — 수치 검증 8항목
       ├─ (수치 통과 시) WanAIValidator.validate() — Gemini Vision 영상 검증
       ├─ (AI 통과 시) _apply_post_compositing()
       │   → bg_remove=True이면 WanBgRemover() → 투명 PNG + APNG + WebM
       │   → bg_remove=False이면 배경 제거 스킵
       │   successful_results.append(result); continue
       └─ (AI 실패 시) AI 조정값 적용 후 다음 시도
            연속 품질 실패 2회 → analyze_with_exclusion()으로 액션 전환

  루프 종료 후: 통계 터미널 출력 → 파일 저장 → VRAM 초기화
               → successful_results[0] 반환 또는 WanResult(success=False)
```

### BG 프롬프트 (전부 중국어 — 항상 강제 추가)
```python
BG_POSITIVE = "纯白色背景，整个动画过程中背景始终保持白色，背景干净无杂质，始终保持恒定的亮度，没有闪烁，画面明亮清晰，边缘锐利，无残影"
    -> 배경과 잔상에 대한 옵션, 깜박임 없음 등
BG_NEGATIVE = "背景变色，背景变暗，背景变黑，背景变灰，背景变黄，背景变紫，背景颜色偏移，非白色背景，黑屏，阴影遮盖，滤镜感，曝光不足，画面闪烁"
    -> 배경색이 변경되는 것을 금지, 필터나 이미지 깜박임 등을 네거티브에 지정
```

---

## 3. 개발 히스토리 (시간순)

---

### 3-1. 기반 구축 — ComfyUI 연동 + 워크플로우
**날짜:** 2026-03-06
**트랜스크립트:** `06-46-00-wan-workflow-api-format-fix`, `07-10-58-wan-gui-format-parser`

#### 구현 내용
- ComfyUI REST API 연동 (`ComfyUIClient`) — 이미지 업로드, 워크플로우 실행, 결과 다운로드
- **GUI format vs API format 차이 발견**
  - ComfyUI WebUI에서 export한 워크플로우는 GUI format (node 위치 정보 포함)
  - API 호출에는 API format 필요 (노드 ID → 입력 매핑 구조)
  - `_gui_workflow_to_api()` 자동 변환 함수 구현
- 워크플로우를 JSON 파일로 저장하고 로드하는 방식 도입

#### 발생한 문제
- `overwrite=true` 미설정으로 같은 파일명에 새 이미지 업로드 시 캐시 이미지 사용됨
  - fish 입력 → 개구리 생성 현상 (ComfyUI 이미지 캐시 문제)
  - 수정: 업로드 API에 `overwrite: true` 파라미터 추가

---

### 3-2. 최초 테스트 — 물고기 + 화질 문제
**날짜:** 2026-03-06  
**트랜스크립트:** `08-57-17-comfyui-fish-preprocessing-quality-fix`, `09-25-32-wan-i2v-prompt-simplification-loop-fix`

#### 발생한 문제: 화질 열화
- 전처리에서 scale=0.65로 이미지를 축소했는데 이것이 PSNR 저하 원인으로 확인됨
- 초기 시스템 프롬프트가 6000자 이상으로 과도하게 복잡 → 모델 혼란

#### 수정 내용
- 프롬프트 6000자 → 2450자로 단순화 (이후에도 계속 개선됨)
- `loop_count=1` 설정 → 루프 경계 열화 방지 (WAN 내부 루프 설정)
- `save_output` 파라미터 정리
- PSNR 측정 도구 추가하여 수치 기반 품질 추적 시작

#### 결과
- 단순화 후 생성 품질 개선 확인
- Q4 모델은 VRAM 한계 확인 → Q3_K_S 모델 사용 결정

---

### 3-3. Validator 구현 + 버그 수정
**날짜:** 2026-03-06
**트랜스크립트:** `05-27-01-wan-validator-md-spec-update`, `05-43-51-bird-wan-validation-failure-analysis`, `10-06-28-wan-i2v-fin-motion-validator-bugs-fix`

#### 최초 Validator 구현 (`wan_validator.py`)
처음 MD 스펙을 기반으로 validator를 구현했으나 스펙과 실제 코드 간 불일치 발견:
- `repeated_motion` peaks 임계값이 스펙과 달랐음
- `return_diff` 수치가 다름
- `center_drift` 항목이 코드에서 누락

수정: MD 스펙 기준으로 validation 로직 재구현

#### 초기 7개 검증 항목
| 항목 | 기준 |
|---|---|
| no_motion | char_motion < min_motion * 0.5 |
| too_slow | char_motion < min_motion |
| too_fast | char_motion > max_motion |
| repeated_motion | peaks > 8 |
| frame_escape | edge_ratio > 0.08 |
| no_return_to_origin | return_diff > 0.20 |
| center_drift | drift > 0.12 |

#### 새(bird) 검증 실패 분석 (8회 실패)
`bird_with_room.png` 8번 실패 분석:
- motion detection 버그: area_contrib 계산으로 오탐
- center_drift false positive 발생
- no_return_to_origin 구조적 문제
- → 당시 fly/wing 특수처리 추가 (이후 제거됨)

#### 물고기 지느러미 모션 문제 (10-06)
- 프롬프트 단순화 후에도 몸통:지느러미 비율 23:1로 악화
- AI 검증에서 no_motion 오판정 발생
- Q3 모델 한계로 미세 지느러미 움직임 생성 어려움 확인
- → 나중에 꼬리 sweep 동작으로 전환 (성공)

---

### 3-4. 프롬프트 전략 전환 — 영어→중국어
**날짜:** 2026-03-06
**트랜스크립트:** `11-52-44-wan-i2v-chinese-prompts-validator-fix`

#### 전환 배경
물고기 attempt01~07까지 운동이 거의 없는 영상이 반복됐다. WAN 2.1이 중국어 데이터로 학습된 모델임을 확인 후 프롬프트를 중국어로 전환.

#### 전환 후 효과
- PSNR: 30.0dB → 32.6dB 향상
- 배경 안정성 눈에 띄게 개선

#### 단계적 전환 과정 (4단계)

**1단계:** positive/negative 프롬프트 중국어 생성 지시 추가 (`简体中文`으로 작성 강제)

**2단계:** `BG_POSITIVE`(배경 흰색 유지 지시) 중국어화 → 배경 안정성 개선

**3단계:** `BG_NEGATIVE`(배경 어두워짐 방지)가 **영어로 남아있던 것**이 새(bird) 테스트에서 배경이 f01부터 RGB 220으로 어두워지는 직접 원인으로 확인 → 중국어화  
(이것이 오랫동안 원인을 몰랐던 배경 어두워짐 버그의 근본 원인)

**4단계:** AI 조정 프롬프트(no_motion 시 수정 지시, return-to-origin 수정 지시 등)도 전부 중국어화

#### 이 세션에서 함께 수정한 내용
- `validator /255 버그` 발견 및 수정 (아래 3-5 참조)
- AI soft 판정 완화
- **하드코딩 규칙 제거**: fish/bird/frog 종류별 분기 코드 제거, Gemini 자유 판단으로 전환
- 5개 스크립트 전체 최종 검증

---

### 3-5. 개구리 테스트 — 점프 사이클 문제
**날짜:** 2026-03-06
**트랜스크립트:** `15-01-30-wan-i2v-validator-fix-frog-motion-issues`, `15-59-40-wan-i2v-frog-jump-cycle-prompt-fix`, `16-41-07-wan-i2v-frog-jump-validator-240p-fix`

#### Validator /255 이중 적용 버그 (이 세션에서 발견)

**증상:** attempt01~02에서 char_motion=0.0이 반복됨. raw_motion=0.003~0.012인데 char_motion=0.0이 나오는 모순.

**원인 추적:**  
`_extract_frames`가 이미 `float32 / 255.0` (0~1 스케일)로 반환하는데  
`_calc_character_motion`과 `_calc_return_diff`에서 또 `/255`를 적용했다.  
0~1 값에 다시 255로 나누면 0~0.004 수준이 되어 threshold(0.04)와 비교 시 항상 no_motion 판정.

**수정:** 두 함수에서 `/255` 제거. `_get_bg_mask` tolerance는 `30/255`로 변환.  
→ 물고기 attempt 통과 확인.

#### 개구리 배/목 호흡 문제

**경위:** 수치 버그 수정 후에도 개구리가 raw_motion=0.001~0.003으로 완전 정지 영상 반복.  
Gemini가 배 호흡, 목 팽창을 반복 선택. Gemini는 `产生清晰的像素位移`(명확한 픽셀 이동)라고 스스로 합리화.

**원인 분석:** 배 호흡은 volumetric change(부피 변화)이지 spatial movement(공간 이동)가 아니다.  
WAN Q3는 색상/픽셀 이동 없이 미세한 형태 변화만으로는 움직임을 생성하지 못한다.

**vision_analyzer 강화:** 공간 이동 동작만 선택하도록 3단계 필터 추가:
- Q1. 루프 가능한가
- Q2. 형태가 유지되는가 (핵심) — 움직이는 동안 부위 구조가 변하지 않는가
  - ✅ PASS: 꼬리 sweep, 귀 쫑긋, 머리 기울임, 다리 뻗기
  - ❌ FAIL: 날개 flap(접힘/펼침 구조 변형), 배 호흡, 목 팽창
- Q3. 범위가 10~40% 이내인가

#### 점프 사이클 처리 (3단계 시도)

**시도 1 — 제자리 뒷다리 굽힘/펼침:**  
→ 뒷다리가 몸통 아래 완전히 접혀 화면에서 거의 안 보이는 상태. WAN이 "없는 부위"를 움직이지 못함.

**시도 2 — 앞발 움직임:**  
→ 수치는 통과하지만 AI가 자연스럽지 않다고 판정.

**시도 3 — 완전한 점프 사이클:**  
→ 수직 이동이 center_drift로 탈락.

**해결:** "강력한 뒷다리를 가진 도약형 생물" SPECIAL CASE 추가.  
완전한 점프 사이클 허용 (수직 이동만, 수평 이동 금지).  
`no_return_to_origin` 판정 완화 (0.20 → 0.40).  
center_drift를 수평 방향만 체크.

**발견된 부작용:** `身体完全静止`(몸통 완전 정지)를 positive에 쓰면 점프 동작과 모순.  
WAN이 혼란을 겪고 아무것도 안 움직이는 영상 생성.  
→ positive에서 "몸통 정지" 표현 사용 금지.

#### 240p 테스트
240×240 해상도 테스트 실시. 속도 확인 후 480×480으로 복귀.
속도는 빠르지만 품질이 보장 되지 않음,
현재 사용중인 Wan-AI/Wan2.1-I2V-14B-480P를 GGUF 포맷으로 직접 변환한 모델 파일(wan2.1-i2v-14b-480p-Q3_K_S.gguf) 사용
wan2.1   - WAN 버전 2.1
i2v      - Image-to-Video (이미지→영상)
14b      - 파라미터 140억개
480p     - 480P 해상도용으로 학습된 모델
Q3_K_S   - 3비트 양자화 (Small 변형)
.gguf    - GGUF 포맷 (ComfyUI-GGUF 노드로 로드)

480P 해상도에 최적화 되어 있음
---

### 3-6. pingpong 도입 + AI Validator 비디오 직접 피드
**날짜:** 2026-03-06
**트랜스크립트:** `18-45-41-wan-i2v-pingpong-validator-fix`, `18-59-50-wan-i2v-ai-validator-video-direct-feed`

#### pingpong 도입

**경위:** 단순 loop 방식에서 영상 끝→시작 이음새가 튀는 현상 발생.

**해결:** `pingpong=True` — 순방향 재생 후 역방향 재생. 이음새 자연스럽게 연결.

**pingpong에서 발생한 AI Validator 오판정:**  
pingpong 구조에서 `MIDDLE(50%)`이 `FIRST(0%)`와 유사한 것은 정상이다.  
AI validator가 이 구조를 몰랐을 때 MIDDLE이 FIRST와 비슷하다는 이유로 no_motion 오탐.

**수정:** AI validator에 pingpong 구조 명시, 모션 판단은 반드시 FIRST vs QUARTER(25%) 비교로.

#### 분류 기반 코드 전체 제거 (이 세션)
`is_jump`, `is_fly`, `is_swim` 분류 플래그 전체 제거.  
통일된 기준 적용 — 모든 캐릭터에 동일한 검증.
특정 모션에 종속된 분류 플래그 값 제거

#### AI Validator: 프레임 추출 → 비디오 직접 전달

**이전 방식:** mp4에서 3프레임 추출 → Gemini에게 이미지 전달  
**문제:** Gemini가 정지 이미지로 모션을 판단하기 어려움

**수정:** mp4 파일 자체를 base64로 인코딩 → Gemini에게 직접 전달  
원본 이미지(스프라이트) + mp4 함께 전달  
→ Gemini가 실제 영상을 보고 모션 품질 직접 판단 가능

#### 개구리 attempt07 최고 품질 달성
비디오 직접 피드 적용 후 개구리 점프 attempt07이 최고 품질로 확인.  
이후 AI 조정 누적으로 품질 저하 발생 → 조정 누적 문제 확인.

---

### 3-7. 물고기 APNG 완성 + frame_escape 오판정 수정
**날짜:** 2026-03-07
**트랜스크립트:** `06-07-49-wan-i2v-fish-motion-analysis`, `09-01-12-wan-i2v-fish-apng-fixes`

#### frame_escape 오판정 — 노란 필터 문제

**증상:** 물고기 attempt01~03에서 frame_escape로 반복 실패. 실제로는 프레임 밖으로 나가지 않음.

**원인:** WAN이 생성한 영상의 배경에 노란 필터가 생기면서 배경 diff가 0.19까지 올라감.  
`_calc_edge_ratio`의 tolerance가 0.15여서 배경 픽셀 전체가 "비배경"으로 인식 → edge_ratio=1.0.

**수정:** tolerance 0.15 → 0.25

#### pingpong 이음새 validator 버그
pingpong 영상의 마지막 프레임이 시작과 유사 → no_return_to_origin 오탐  
pingpong 구조를 validator에서도 인식하도록 수정

#### no_motion seed 재시도 로직 추가
no_motion 판정 시 같은 프롬프트로 seed 변경해 재시도하는 옵션 추가  
(단순 운이 나쁜 경우와 실제 생성 불가 경우를 구분)

#### AI Validator 중국어 프롬프트 수정
AI 검증 결과 전달에 사용되는 내부 텍스트도 중국어로 통일

#### APNG disposal=2 수정
`disposal=2`: 각 프레임 후 투명으로 초기화  
→ 브라우저에서 APNG 재생 시 이전 프레임 잔상 방지

#### WanBgRemover tolerance 조정
**증상:** 물고기 APNG에서 배경 회색 잔상이 남는 현상.

**원인:** WAN이 생성한 배경이 실제로는 흰색(255)이 아니라 회색(~RGB 206~217)으로 나오는 경우.  
tolerance=35이면 `diff(255-206)=49 > 35`이 되어 배경 픽셀이 제거 안 됨.

**수정:** tolerance=35 → **50**

#### APNG 뷰어 HTML 생성
생성된 APNG를 확인하기 위한 debug_viewer.html 생성

---

### 3-8. 새(bird) 테스트 — BG 문제 + 날개 flap 10회 실패
**날짜:** 2026-03-07
**트랜스크립트:** `10-35-10-wan-i2v-bird-bg-fix-session`, `11-38-40-wan-i2v-bird-motion-quality-session`

#### 배경 어두워짐 문제 (BG_NEGATIVE 영어 잔존)

**증상:** 새 테스트에서 attempt01, 06, 07, 08이 배경이 f01부터 RGB 220으로 어두워짐.

**원인 발견:** `BG_NEGATIVE`가 영어로 남아있었음.  
```
# 영어 상태 (버그):
BG_NEGATIVE = "dark background, gray background..."

# 수정 후 (중국어):
BG_NEGATIVE = "背景变色，背景变暗，背景变黑，背景变灰..."
```
이것이 배경 어두워짐의 직접 원인이었음. 중국어로 변환 후 해결.

#### positive 누적 제한 추가
AI가 매 attempt마다 positive에 내용을 append → attempt07에서 300자 초과  
서로 충돌하는 지시가 생기기 시작  
→ positive 150자 초과 시 초기값 + 새 조정으로 리셋 규칙 추가

#### 날개 flap 10회 실패 분석

새 테스트에서 날개 flap을 10회 이상 시도했지만 전부 실패.

**실패 원인 분석:**
| 캐릭터 | 동작 | 결과 | 이유 |
|---|---|---|---|
| 물고기 | 꼬리 sweep | ✅ 성공 | 꼬리 형태 유지하며 위치만 이동 |
| 개구리 | 점프 | ✅ 성공 | 전체가 rigid unit으로 이동 |
| 새 | 날개 flap | ❌ 10회+ 실패 | 날개가 접히고 펼쳐지면서 형태 자체가 변형 |

날개 flap은 3D 구조 변형(접힘/펼침)을 동반한다.  
2D 픽셀 기반 모델(WAN Q3)이 날개의 3D 구조 변형을 텍스처 일관성 있게 생성하는 것은 근본적으로 어렵다.

**결정:** 날개 flap 시도 완전 포기.  
**전략 전환:** "LOWEST DIFFICULTY(가장 쉬운 것)" → "MOST NATURAL motion(가장 자연스러운 것) + WAN 생성 가능성 필터"

**AI Validator 판정 기준 범용화:**
- 초기: `wing/fin/feathers` 예시가 박혀 있어서 새/물고기에만 적용한다고 오해 가능
- 수정: 모두 `the moving part` / `fixed parts`로 추상화 → 어떤 모션에도 동일 적용

**5가지 실패 패턴 명문화:**  
새 attempt07에서 AI가 "성공"이라고 인정하면서도 FAIL을 낸 사례 발생.  
Gemini에게 직접 영상을 분석시켜 지적된 내용을 기준으로 추가:
- **FLICKERING**: 움직이는 부위가 부드러운 호 대신 프레임마다 급격히 다른 형태로 튀는 경우
- **GHOSTING**: 이전 프레임 이미지가 모션 경로를 따라 블리딩
- **TEXTURE RECONSTRUCTION**: 텍스처(깃털/비늘 등)가 매 프레임 재생성
- **PART INDEPENDENCE**: 고정되어야 할 부위의 형태·텍스처가 변형 (1~2px 미세 진동은 PASS)
- **INCOHERENT MOTION RHYTHM**: pendulum/sine curve가 아닌 랜덤 가속/감속으로 떨리는 느낌

**vision_analyzer STEP 2~6 전면 범용화:**
- 새/물고기/개구리 특수처리 제거
- 어떤 이미지든 동일한 VISION_SYSTEM_PROMPT 적용
- 계속 동일한 이미지를 사용하다 보니 계속, 새/물고기/개구리의 이미지에만 특화된 기능들이 추가되는 문제로 범용적으로 사용할 수 있게 제거 처리

---

### 3-9. 마스킹 파이프라인 도입 + 후처리 합성 전환
**날짜:** 2026-03-07
**트랜스크립트:** `12-46-23-wan-i2v-masking-pipeline-session`, `13-57-11-wan-i2v-masking-postcomp-session`

#### WanMaskGenerator 신규 구현

**목적:** moving_zone(움직이는 부위의 bbox)에만 WAN 모션을 적용하고 나머지는 원본 이미지를 유지  
**구현:** moving_zone bbox → 흑백 마스크 PNG 생성  
- 전체 흰색 = fixed (원본 유지)
- moving zone = 검정 (WAN 생성 사용)
- 경계: GaussianBlur (이미지 크기의 3%) → 자연스러운 경계

#### SetLatentNoiseMask 비호환 발견

`SetLatentNoiseMask` 노드로 마스크를 WAN에 직접 주입 시도 → **실패**

**원인:** WAN은 video latent(5D 텐서: batch × channels × frames × height × width)를 사용하는데  
SetLatentNoiseMask는 이미지 latent(4D: batch × channels × height × width)를 기대한다.  
차원 불일치로 실행 자체가 안 됨.

**전환 결정:** 마스크를 ComfyUI 내부에서 사용하지 않고, **후처리 합성(Post-Compositing)** 방식으로 변경.

#### `_apply_post_compositing()` 구현

```python
# 픽셀 레벨 합성: 마스크 기반으로 WAN 생성 프레임과 원본 이미지 합성
# 흰색 마스크 영역: 원본 이미지 유지
# 검정 마스크 영역: WAN 생성 프레임 사용
composited = mask * original + (1-mask) * generated
```

→ 고정 부위는 픽셀 레벨에서 원본을 그대로 사용하므로 ghosting/flickering 원천 차단

#### 이전 세션의 fly 특수처리 제거 (이 세션)
fly 관련 특수처리 코드 제거, SOFT_ISSUES 재정리

---

### 3-10. 합성 순서 버그 + 액션 전환 로직 추가
**날짜:** 2026-03-08
**트랜스크립트:** `03-16-51-wan-i2v-compositing-motion-fix`, `04-06-53-wan-i2v-validator-action-switch-manual`

#### 합성→검증 순서 버그 발견

**버그:** `_apply_post_compositing()`를 수치 검증 **전에** 실행하고 있었음.

**문제:** 합성된 영상은 원본 정보가 반영되어 수치 검증 결과가 왜곡됨.  
실제 WAN이 생성한 모션을 수치로 측정해야 하는데 합성 후 측정하면 오탐.

**수정:** 검증(수치 + AI) → 통과 후 합성 순서로 변경.

```python
# 수정 전 (버그):
composited = _apply_post_compositing(raw_video)
result = validate(composited)

# 수정 후 (정상):
result = validate(raw_video)
if result.passed:
    final = _apply_post_compositing(raw_video)
```

#### vision_analyzer: identity motion 우선 선택
동작 선택 시 가장 자연스러운 "IDENTITY MOTION" 을 우선시하는 방식으로 전환.

#### 연속 품질 실패 시 액션 전환 로직 추가

**문제:** 특정 동작(예: 날개 flap)이 반복 실패할 때 같은 동작만 계속 재시도함.

**수정:** 연속 품질 실패 3회 → `analyze_with_exclusion()` 호출  
→ 실패한 액션을 명시하고 다른 동작을 선택하도록 요청

```python
def analyze_with_exclusion(self, image, excluded_actions: list[str]):
    # excluded_actions를 명시하여 다른 동작 선택 강제
    # temperature=0.5 (약간 창의적으로)
```

#### 수동 선택 영상 처리 메서드 추가
`process_manual_selection(video_path)` — 특정 attempt 결과를 수동으로 선택해 후처리하는 메서드

---

### 3-11. 수치 검증 대규모 오탐 수정
**날짜:** 2026-03-08
**트랜스크립트:** `05-22-26-wan-i2v-validator-frame-quality-fixes`, `06-37-57-wan-i2v-validator-bg-fix-session`, `08-27-12-wan-i2v-validator-generality-fix`

#### area_contrib 제거
기존 `area_contrib` 계산이 오탐의 원인이었음 → 제거.

#### peaks 상향
`repeated_motion` peaks 임계값: 8 → **12** (pingpong 기준으로 상향)  
pingpong 영상에서는 자연스럽게 peaks가 높아지기 때문.

#### too_fast AND 조건 추가
```python
# 수정 전:
too_fast = char_motion > max_motion

# 수정 후 (오탐 방지):
too_fast = char_motion > max_motion AND raw_motion > min_motion * 0.5
```
raw_motion 교차 검증으로 빠른 배경 변화를 캐릭터 모션으로 오탐하는 경우 방지.

#### frame_count 동적 결정 도입
기존: 고정 frame_count  
변경: Vision Analyzer가 SHORT/MEDIUM/LONG 중 선택
- SHORT: 17~21 프레임
- MEDIUM: 25~33 프레임
- LONG: 33~49 프레임

#### AI Validator 신체 소실/회색 박스 감지 강화 (05:22)
영상에서 캐릭터가 사라지거나 회색 박스로 대체되는 경우 감지 추가

#### JSON 파싱 복구 3단계
Gemini 응답에서 JSON 파싱 오류 발생:
```
1단계: 정상 파싱
2단계: reason 필드 제거 후 재파싱 (reason에 따옴표가 들어있는 경우)
3단계: passed/issues 필드만 regex 추출
```

#### bg_drift 단위 버그 수정
배경 drift 계산에서 단위 변환 버그 발견 및 수정.

#### frame_escape 오탐 방지
기존 frame_escape 로직이 배경 변색을 이탈로 오탐하는 추가 케이스 발견 및 수정.

#### AI 판정 기준 완화
- no_motion 판정 기준 완화 (수치 validator 통과 후 AI도 no_motion 이중 판정 방지)
- unnatural_movement 판정 완화
- SOFT_ISSUES 확장

#### White Anchor 전처리 추가
`_white_anchor()` 함수 추가:
- 배경 순백색 강제 (RGB 255,255,255)
- 경계 픽셀 선명화
- 검정 배경 이미지도 흰 배경으로 자동 변환 → 별도 처리 불필요

#### 배경만 변색 시 통과 로직
AI validator:
- Category A (배경만 변색, 캐릭터 정상): PASS — bg_remover가 후처리로 제거
- Category B (전체 밝기 이동, 캐릭터도 어두워짐): FAIL

#### 범용성 검토 완료
모든 검증 로직이 물고기/새/개구리 이외 이미지에도 적용 가능한지 검토 완료.

---

### 3-12. Ghosting 감지 강화 + 프롬프트 누적 제거
**날짜:** 2026-03-08 
**트랜스크립트:** `14-14-05-wan-i2v-ghosting-motion-fix`

#### GHOSTING을 AI Validator 최상단으로 이동

**경위:** ghosting이 발생한 영상이 다른 항목 검사 중간에 통과하는 경우 발생.

**수정:**
```
[이전 순서]
1. SPEED
2. RETURN TO ORIGIN
3. FRAME ESCAPE
4. NATURALNESS (GHOSTING 포함)
5. CHARACTER CONSISTENCY

[수정 후 순서]
1. GHOSTING (ZERO TOLERANCE — check this FIRST)  ← 최상단 이동
2. SPEED
3. RETURN TO ORIGIN
...
```

**GHOSTING 두 형태 명시:**
- Form 1 (OUTLINE GHOST): 움직이는 부위 윤곽에 반투명 잔상
- Form 2 (BODY DRIFT GHOST): 고정 몸통이 미세하게 이동하며 두 개로 겹쳐 보임
- 고정 몸통이 soft/translucent/doubled이면 무조건 FAIL

#### `ghost_score` 수치 메트릭 추가 (`wan_validator.py`)

```python
GHOST_THRESHOLD = 0.005  # 0.5%

def _calc_ghost_score(self, frames, bg_mask):
    """
    첫 프레임 배경 픽셀 위치에 이후 프레임에서
    캐릭터 평균색과 80/255 이내 픽셀이 나타나는 비율
    """
    # 첫 프레임 배경 픽셀 위치 기억
    # 이후 프레임에서 해당 위치에 캐릭터 색상 출현 비율 계산
    # 0.5% 이상 = ghosting FAIL
```

#### 프롬프트 누적 방지 규칙 강화

**최악의 사례:**
`后腿轻轻弯曲和伸展`(뒷다리 살살 굽힘)이 positive에 쓰인 후,  
AI 조정 과정에서 이 표현을 negative에도 추가.  
결과: 뒷다리 굽힘 동작 자체가 억제되어 완전 정지 영상 생성.

**규칙 추가:**
- positive 150자 초과 시 초기값 + 새 조정으로 **리셋**
- negative 200자 초과 시 동일하게 **리셋**
- **STEP G: positive에서 쓴 표현을 negative에 절대 복사 금지**

#### Negative 우선 원칙 확립

**발견:** WAN 특성상 positive로 동작을 유도하는 것보다 negative로 원치 않는 동작을 막는 것이 더 직접적이고 효과적이다.

**적용:** `naturalness/character` 이슈의 조정은 `NEGATIVE IS PRIMARY`로 명시.  
positive는 보조 수단.

#### Vision Analyzer: MOST NATURAL motion 방향 전환
- "LOWEST DIFFICULTY" 표현 삭제
- "MOST NATURAL motion that WAN can generate" 로 변경
- 날개 운동 우선 (새), 10~40% 가시범위 강제

---

### 3-13. 암전 방지 + SOFT_ISSUES 수정 + 테스트 캐릭터 생성
**날짜:** 2026-03-08
**트랜스크립트:** `16-31-16-wan-i2v-ghosting-blackout-test-chars`

#### 암전/필터 방지 BG 프롬프트 강화

기존 BG_POSITIVE/NEGATIVE에 암전 관련 키워드 추가:
```python
# BG_POSITIVE에 추가:
"始终保持恒定的亮度，没有闪烁，画面明亮清晰，边缘锐利，无残影"

# BG_NEGATIVE에 추가:
"背景变灰，背景变黄，背景变紫，非白色背景，黑屏，阴影遮盖，滤镜感，曝光不足，画面闪烁"
```

#### SOFT_ISSUES에서 no_motion 제거

**이전:** `SOFT_ISSUES = {"background_color_change", "no_motion"}`  
→ 수치 validator 통과 후 AI가 no_motion만 판정하면 PASS로 처리했음

**문제:** AI가 no_motion을 판정한다는 것은 육안으로도 정지처럼 보인다는 의미.  
이런 영상을 PASS시키면 실제로 움직이지 않는 영상이 최종 결과물이 됨.

**수정:** `SOFT_ISSUES = {"background_color_change"}`  
background_color_change만 soft인 이유: bg_remover가 후처리로 배경을 제거하기 때문.

#### MAX_RETRIES=5로 임시 변경
테스트 속도를 위해 10 → **5** 임시 변경. 운영 시 반드시 10으로 복구 필요.

---

### 3-14. successful_results 루프 수정 + 이미지 분리
**날짜:** 2026-03-08
**트랜스크립트:** `17-45-39-wan-i2v-allretries-charcrops-session`, `17-47-22-wan-i2v-sprite-split-allretries-handoff`, `17-48-57-wan-i2v-backend-retries-code-summary`

#### wan_backend.py: successful_results 루프 수정 (핵심 수정)

**문제:**
```python
# 버그: 성공 즉시 return → 첫 성공에서 루프 완전 종료
if ai_result.passed:
    return WanResult(success=True, ...)
```

**의도:** MAX_RETRIES 횟수 전부를 생성하여 나중에 비교/선택 가능하게 한다.

**수정 (3곳):**

```python
# ① 루프 시작 전 (약 1160번 줄) — 리스트 초기화
successful_results: list[WanResult] = []

# ② 성공 시 return → continue 변경 (약 1257~1272번 줄)
# 수정 전:
return WanResult(success=True, video_path=apng_path, ...)

# 수정 후:
successful_results.append(WanResult(success=True, video_path=apng_path, ...))
continue

# ③ 루프 종료 후 (약 1468~1476번 줄) — 첫 성공 반환
if successful_results:
    return successful_results[0]
return WanResult(success=False, analysis=analysis, attempts=MAX_RETRIES)
```

→ 모든 attempt를 생성하고, 성공한 결과들 중 첫 번째를 반환.

---

## 4. 검증 시스템 발전 과정

### 수치 Validator (`wan_validator.py`) 진화

| 시점 | 변경 내용 |
|---|---|
| 초기 | 7개 항목 구현 (no_motion, too_slow, too_fast, repeated_motion, frame_escape, no_return_to_origin, center_drift) |
| 03-06 | /255 이중 적용 버그 수정 → 물고기 통과 |
| 03-06 | frame_escape tolerance 0.15 → 0.25 (노란 필터 오탐 방지) |
| 03-06 | 점프 생물 no_return_to_origin 완화 (0.20 → 0.40) |
| 03-06 | center_drift 수평만 체크 (점프 생물 수직 이동 허용) |
| 03-07 | pingpong 이음새 validator 버그 수정 |
| 03-08 | area_contrib 제거 |
| 03-08 | peaks 8 → 12 (pingpong 기준) |
| 03-08 | too_fast AND 조건 추가 |
| 03-08 | frame_count 동적 결정 |
| 03-08 | bg_drift 단위 버그 수정 |
| 03-08 | **ghost_score 신규 추가** (GHOST_THRESHOLD=0.005) |
| 03-08 | 배경색 자동 감지 (첫 프레임 테두리 픽셀 평균) |

### AI Validator (`wan_ai_validator.py`) 진화

| 시점 | 변경 내용 |
|---|---|
| 초기 | 프레임 추출 (3장) → Gemini 전달 |
| 03-06 | mp4 직접 전달로 전환 (원본 이미지 + 영상) |
| 03-06 | pingpong 구조 명시 (MIDDLE ≈ FIRST는 정상) |
| 03-06 | 분류 기반 코드 (is_jump/fly/swim) 전체 제거 |
| 03-07 | 5가지 실패 패턴 명문화 (FLICKERING/GHOSTING/TEXTURE RECONSTRUCTION/PART INDEPENDENCE/INCOHERENT MOTION RHYTHM) |
| 03-07 | 판정 기준 범용화 (`the moving part`/`fixed parts`로 추상화) |
| 03-08 | 신체 소실/회색 박스 감지 추가 |
| 03-08 | JSON 파싱 복구 3단계 추가 |
| 03-08 | AI 판정 기준 완화 (no_motion, unnatural_movement) |
| 03-08 | **GHOSTING ZERO TOLERANCE 최상단 이동** |
| 03-08 | Form 1 (OUTLINE GHOST) + Form 2 (BODY DRIFT GHOST) 명시 |
| 03-08 | Background color: Category A/B 구분 (A는 PASS) |

### 현재 수치 Validator 8항목
```
no_motion         char_motion < min_motion * 0.5
too_slow          char_motion < min_motion
too_fast          char_motion > max_motion AND raw_motion > min_motion*0.5
repeated_motion   peaks > 12 (pingpong 기준)
frame_escape      edge_ratio > 0.08, tolerance=0.25
no_return_to_origin  return_diff > 0.40
center_drift      drift > 0.12
ghosting          ghost_score > 0.005
```

---

## 5. 프롬프트 개선 과정

### Vision Analyzer (`wan_vision_analyzer.py`) 프롬프트 진화

| 시점 | 변경 내용 |
|---|---|
| 초기 | 6000자 복잡한 시스템 프롬프트 |
| 03-06 | 6000자 → 2450자 단순화 |
| 03-06 | 영어 → 중국어 전환 (positive/negative) |
| 03-06 | MOVING PART ISOLATION 원칙 추가 |
| 03-06 | 배 호흡/목 팽창 금지 (volumetric vs spatial 구분) |
| 03-06 | 3단계 WAN 필터 추가 (Q1 루프/Q2 형태 유지/Q3 범위) |
| 03-06 | 점프 생물 SPECIAL CASE 추가 |
| 03-06 | positive에서 "몸통 정지" 표현 금지 |
| 03-07 | 분류 기반 특수처리 전체 제거 |
| 03-07 | STEP 2~6 전면 범용화 |
| 03-07 | moving_zone bbox 추가 (마스크 생성용) |
| 03-08 | identity motion 우선 선택 |
| 03-08 | "LOWEST DIFFICULTY" → "MOST NATURAL motion + WAN 필터" |
| 03-08 | analyze_with_exclusion() 추가 |

### AI 조정 프롬프트 누적 방지 (wan_backend.py)

**문제의 발전 과정:**
1. 매 attempt마다 AI가 positive/negative에 내용 append
2. attempt07에서 positive 300자 초과
3. 서로 충돌하는 지시 발생
4. **최악 사례:** positive 표현이 negative에 복사 → 동작 자체 억제

**최종 규칙:**
```python
# positive 150자 초과 시 리셋
if len(current_positive) > 150:
    current_positive = initial_positive + new_adjustment

# negative 200자 초과 시 리셋
if len(current_negative) > 200:
    current_negative = initial_negative + new_adjustment

# STEP G: positive 표현을 negative에 절대 복사 금지
```

### 현재 VISION_SYSTEM_PROMPT 단계 (완성본)
```
STEP 1: IDENTIFY THE SUBJECT — 외형·색상·부위
STEP 2: CHOOSE THE ACTION — MOST NATURAL motion + 3단계 WAN 필터
         Q1. 루프 가능한가
         Q2. 형태가 유지되는가 (핵심)
         Q3. 범위가 10~40% 이내인가
         SPECIAL CASES: 도약형 생물 → 완전한 점프 사이클 + 수직 이동만
STEP 3: ISOLATE MOVING PARTS — 최소 부위, 나머지 전신 고정
STEP 4: MOVING ZONE BBOX — 상대좌표 [x1,y1,x2,y2], 20% 여유
STEP 5: FRAME RATE — 8~24fps
STEP 6: WRITE PROMPTS — 중국어. positive 60단어 이내, negative 40단어 이내
STEP 7: MOTION THRESHOLDS — min_motion·max_motion·max_diff
STEP 8: FRAME COUNT — SHORT 17~21 / MEDIUM 25~33 / LONG 33~49
```

### 금지 단어 (whole-body 회전/이동 유발)
```
旋转, 转身, 扭动, 摇摆身体, 全身运动
```

---

## 6. 핵심 설계 결정 배경 (Why)

### 6-1. 영어 → 중국어 프롬프트
WAN 2.1은 중국어 데이터로 학습된 모델. PSNR 30.0dB → 32.6dB 향상. BG_NEGATIVE가 영어로 남아있던 것이 배경 어두워짐(RGB 220)의 직접 원인이었음.

### 6-2. steps=30 → 20
생성 시간 33% 단축. frames는 줄이면 pingpong 복귀 문제 악화 → 유지.

### 6-3. pingpong=True
이음새 완화. MIDDLE(50%)이 FIRST(0%)와 유사한 것은 정상.

### 6-4. 하드코딩 분기 제거
fish/bird/frog 분기 → 미지 이미지 대응 불가. 전체 제거, Gemini 자유 판단.

### 6-5. 배 호흡·목 팽창 금지
volumetric change는 WAN 생성 불가. spatial movement만 허용.

### 6-6. 새 날개 flap 10회 실패 → 저난이도 원칙
날개 flap은 3D 구조 변형 동반 → 2D 픽셀 모델 생성 불가. "MOST NATURAL + WAN 생성 가능성 필터" 원칙.

### 6-7. 개구리 점프 처리
뒷다리 굽힘(보이지 않음) → 앞발(부자연스러움) → 완전한 점프 사이클. 몸통 정지 표현이 점프와 모순 → positive에서 금지.

### 6-8. AI 조정 누적 문제
positive 표현이 negative에 복사되어 동작 억제. positive 150자/negative 200자 초과 시 리셋. STEP G: positive 표현 negative 복사 절대 금지.

### 6-9. Negative 우선 원칙
WAN은 positive 유도보다 negative 차단이 더 효과적. naturalness/character 이슈 = NEGATIVE IS PRIMARY.

### 6-10. Validator /255 이중 적용 버그
`_extract_frames`가 이미 0~1 스케일 반환 → 두 함수에서 또 /255 적용 → char_motion=0.0 오판정. 제거.

### 6-11. frame_escape tolerance 오탐
배경 노란 필터(diff~0.19) > tolerance(0.15) → 날개 이탈로 오탐. 0.15 → 0.25.

### 6-12. WanBgRemover tolerance=50
WAN 생성 배경이 회색(206~217)으로 나오는 경우. tolerance=35이면 diff=49>35로 제거 안 됨. → 50.

### 6-13. GHOSTING 최상단 이동
Ghosting 발생 영상이 다른 항목 검사 중 통과. ZERO TOLERANCE — 가장 먼저. 수치에도 ghost_score 추가.

### 6-14. 5가지 실패 패턴 명문화
AI가 "성공"이라고 인정하면서도 FAIL. Gemini가 직접 지적한 내용을 기준으로 추가. `the moving part`/`fixed parts`로 추상화.

### 6-15. SOFT_ISSUES에서 no_motion 제거
AI no_motion = 육안으로도 정지처럼 보임. SOFT_ISSUES = {"background_color_change"}만 유지.

### 6-16. successful_results 루프 수정
성공 즉시 return → 첫 성공에서 루프 종료. 수정: return → append + continue. 모든 attempt 생성 후 첫 번째 성공 반환.

### 6-17. SetLatentNoiseMask 비호환 → 후처리 합성
WAN video latent는 5D 텐서. SetLatentNoiseMask는 4D 기대 → 차원 불일치로 실패. 픽셀 레벨 후처리 합성으로 전환.

### 6-18. 합성→검증 순서 버그
합성된 영상으로 수치 측정 → 원본 모션 왜곡됨. 검증 먼저, 통과 후 합성으로 순서 변경.

---

## 7. 최종 파일 현황

### 스크립트 파일 (6개)
로컬 경로: `~/anim_pipeline/image_pipeline/sprite_gen/`
Git 레포: https://github.com/solidSnakesado/Create_image_motion/tree/dev

| 파일 | 줄수 | 역할 |
|---|---|---|
| `wan_backend.py` | 2197 | 메인 오케스트레이터. 전체 흐름 제어 + VRAM 초기화/모니터링 + 검증 통계 + 로그 파서 |
| `wan_ai_validator.py` | 541 | Gemini Vision으로 영상 품질 검증 |
| `wan_vision_analyzer.py` | 491 | Gemini Vision으로 이미지 분석 + 프롬프트/수치/pingpong/bg_type 결정 |
| `wan_validator.py` | 708 | 수치 기반 8항목 검증 |
| `wan_mask_generator.py` | 104 | moving_zone → 흑백 마스크 PNG |
| `wan_bg_remover.py` | 254 | flood fill 배경 제거 → APNG + WebM |

---

## 8. 2주차 개발 히스토리
> 기간: 2026-03-09 ~ 2026-03-10
> 목적: Git 레포 구성 + 새 환경 셋업 검증 + VRAM 안정화 + 검증 통계 시스템 추가

---

### 8-1. Git 레포 셋업 + 코드 업로드
**날짜:** 2026-03-09
**레포:** https://github.com/solidSnakesado/Create_image_motion/tree/dev

#### Git 업로드 대상 파일 정리
WAN I2V 파이프라인 코드 6개 + `__init__.py` 2개 + `.env.example` + `requirements.txt`를 `dev` 브랜치에 업로드.

**Git에 올린 파일:**
| 파일 | 역할 |
|---|---|
| `image_pipeline/__init__.py` | 패키지 초기화 |
| `image_pipeline/sprite_gen/__init__.py` | 패키지 초기화 |
| `image_pipeline/sprite_gen/wan_backend.py` | 메인 오케스트레이터 |
| `image_pipeline/sprite_gen/wan_vision_analyzer.py` | Gemini Vision 분석 |
| `image_pipeline/sprite_gen/wan_validator.py` | 수치 검증 8항목 |
| `image_pipeline/sprite_gen/wan_ai_validator.py` | AI 영상 검증 |
| `image_pipeline/sprite_gen/wan_mask_generator.py` | 마스크 생성 |
| `image_pipeline/sprite_gen/wan_bg_remover.py` | 배경 제거 |
| `.env.example` | 환경변수 템플릿 |
| `requirements.txt` | Python 의존성 |
| `workflows/wan21_native_i2v_1.json` | ComfyUI 워크플로우 |
| `WSL_FULL_SETUP_v2.md` | WSL 셋업 가이드 |

**Git에 올리지 않는 것:**
- ComfyUI 자체 (`git clone`으로 설치)
- ComfyUI `main.py` (ComfyUI 자체 코드, 우리가 수정하지 않음)
- 모델 파일 (`.gguf`, `.safetensors` — 수 GB 바이너리)
- `animVenv/`, `venv/` (가상환경)
- `.env` (API 키 포함)

#### Git 셋업 이슈 해결
- `git config --global user.email/name` 설정 필요 (최초)
- `aiacademy8th` 계정 자격증명 충돌 → `git remote set-url`로 username 지정
- GitHub 비밀번호 → PAT(Personal Access Token) 필요
- `git config --global credential.helper store`로 토큰 재입력 방지

---

### 8-2. WSL 새 환경 셋업 + 검증
**날짜:** 2026-03-09
**환경:** Ubuntu 24.04 on WSL2 (snake2 계정)

#### 셋업 가이드 문서 작성 및 검증
`WSL_FULL_SETUP_v2.md` 작성. 실제 설치 과정에서 발견된 이슈를 반영하여 3번 수정.

#### 설치 중 발생한 이슈와 해결

**GPU 접근 차단:**
```
Failed to initialize NVML: GPU access blocked by the operating system
```
→ `wsl --shutdown` 후 재시작으로 해결

**CUDA Toolkit 12-4 설치 실패:**
```
nsight-systems-2023.4.4 : Depends: libtinfo5 but it is not installable
```
→ Ubuntu 24.04에서 `libtinfo5` 미제공. **cuda-toolkit-12-6**으로 변경

**dpkg lock 에러:**
`unattended-upgrades` 자동 업데이트 프로세스 충돌 → `sudo kill <PID>` 후 재시도

**huggingface-cli 실행 불가:**
`huggingface-hub` 1.6.0에서 CLI 모듈 구조 변경 → Python 코드 직접 호출 방식으로 전환:
```python
from huggingface_hub import hf_hub_download
hf_hub_download('repo', 'file', local_dir='dir/')
```

**모델 다운로드 경로 문제:**
HuggingFace에서 `split_files/` 하위 폴더 생성 → `mv`로 올바른 위치로 이동 필수

---

### 8-3. RTX 50XX PyTorch 호환성 해결
**날짜:** 2026-03-09

#### 문제: sm_120 호환성 에러
```
RTX 5070 Ti Laptop GPU with CUDA capability sm_120 is not compatible
current PyTorch install supports CUDA capabilities sm_50 ~ sm_90
```
RTX 50XX(Blackwell)는 CUDA capability sm_120. PyTorch cu126(stable)은 sm_90까지만 지원.

#### 해결 과정 (3단계)

**1단계: cu126 → cu128 (stable)**
sm_120 에러 해결. 하지만 생성 속도가 기존 대비 ~50% 느림.
원인: `WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations`
→ `comfy_kitchen` 백엔드 triton/cuda가 모두 disabled, eager만 사용

**2단계: cu128 stable → cu128 nightly**
기존 환경 확인 결과 `2.12.0.dev20260303+cu128` (nightly) 사용 중.
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
→ 기존 환경과 동일한 속도 복원

#### GPU 아키텍처별 PyTorch 설치 정리
| GPU | 아키텍처 | PyTorch |
|---|---|---|
| RTX 50XX | Blackwell sm_120 | cu128 **nightly** 필수 |
| RTX 40XX | Ada Lovelace sm_89 | cu126 stable 이상 |
| RTX 30XX | Ampere sm_86 | cu124 stable 이상 |

---

### 8-4. ComfyUI 버전 고정 (VRAM 행 방지)
**날짜:** 2026-03-09 ~ 03-10

#### 문제: 생성 중 85%에서 행(hang)
새 환경(ComfyUI 0.16.4)에서 생성 진행률 85% (17/20 steps)에서 멈춤.
`nvidia-smi` 확인: VRAM 9572MB/12227MB 사용, GPU-Util 0%.

#### 원인 분석
- ComfyUI v0.16.4의 메모리 관리 방식이 12GB VRAM에서 문제
- VAE 디코드 단계에서 VRAM 부족 → 모델 오프로드 과정에서 교착 상태

#### 해결
기존 환경의 ComfyUI 커밋 확인 → 동일 커밋으로 고정:
```bash
git checkout eb011733  # v0.15.1 + 패치
```
→ 85% 행 문제 해결, 정상 생성 확인 (20/20 steps, ~5분)

#### DB 스키마 충돌
v0.16.4에서 생성된 DB가 v0.15.1과 비호환:
```
Can't locate revision identified by '0002_merge_to_asset_references'
```
→ `rm ~/ComfyUI/user/comfyui.db ~/ComfyUI/user/comfyui.db.lock`로 해결

---

### 8-5. VRAM 누수 문제 확인 + 초기화 기능 추가
**날짜:** 2026-03-10

#### 문제: 연속 생성 시 VRAM 누수 → 행 발생
ComfyUI v0.15.1에서도 12~15회 연속 생성 후 행 발생.
두 환경(sado Ubuntu 22.04, snake2 Ubuntu 24.04) 모두 동일 증상.
→ OS/Python 버전 차이가 아닌 **cudaMallocAsync 할당기의 VRAM 누수 누적** 문제.

#### 기존 환경 테스트 결과
4개 이미지 연속 생성 (개구리→코끼리→헬리콥터→여성 포트레이트):
- 개구리: 5/5 실패 (이미지 난이도)
- 코끼리: 5/5 실패 (미세 동작)
- 헬리콥터: 4회 생성 후 **5회째 1800초 타임아웃** (VRAM 행 시작)
- 여성 포트레이트: **5회 전부 1800초 타임아웃** (ComfyUI 완전 멈춤)

#### 해결: ComfyUI `/free` API로 VRAM 초기화
`wan_backend.py`에 두 가지 추가:

**1. `ComfyUIClient.free_memory()` 메서드:**
```python
def free_memory(self) -> None:
    # ComfyUI /free API 호출 → 모델 언로드 + CUDA 캐시 클리어
    # nvidia-smi로 전후 VRAM 사용량 표시
```

**2. `generate()` 루프 종료 후 자동 호출:**
```
이미지1 → 시도1~5 → free_memory() → return
이미지2 → 시도1~5 → free_memory() → return
```

**로그 출력 예시:**
```
[ComfyUI] VRAM 초기화 완료 — 전: 9572MB / 12227MB → 후: 438MB / 12227MB (해제: 9134MB)
```

---

### 8-6. 검증 실패 통계 + 보완 조치 추적 기능
**날짜:** 2026-03-10

#### 추가된 기능: `_ValidationStats` 클래스

**기록하는 항목:**
- 수치 검증 실패 (항목별: no_motion, too_slow, ghosting 등)
- AI 검증 실패 (항목별: unnatural_movement, character_inconsistency 등)
- 수치 실패 시 함께 발생한 AI 이슈 (`ai_issues` 필드 — 별도 추적)
- 성공
- 보완 조치 4종: `seed_retry`, `ai_adjust`, `action_switch`, `soft_pass`

**출력 위치:**
1. 터미널 로그 — 이미지별 비율 + 누적 비율 (이미지 완료 시)
2. 파일 — `outputs/wan/validation_stats.txt` (매 시도 즉시 기록 + 이미지 완료 시 요약 추가)

**파일 저장 방식 (flush 방식):**
- **다음 시도 시작 시 이전 시도를 확정 저장** (`_flush_pending_attempt()`)
  - remedy/ai_issues가 모두 확정된 후에 기록 → 정보 누락 없음
  - 중간 중단(Ctrl+C)해도 직전까지의 시도가 파일에 남음
- **이미지 완료 시 마지막 시도 flush + 요약 추가** (`save_to_file()`)
- 성공 시에도 기록

```
시도 1: record() → pending (remedy/ai_issues 미확정)
  → AI 검증 → _update_last_ai_issues() → _update_last_remedy()
시도 2: record() → 시도1 flush (확정 저장) → 시도2 pending
  → AI 검증 → 업데이트
시도 3: record() → 시도2 flush → 시도3 pending
...
시도 5: record() → 시도4 flush → 시도5 pending
────── 루프 종료 ──────
save_to_file()             → 시도5 flush + 요약 통계 기록
print_image_summary()      → 터미널 출력
print_cumulative_summary()  → 터미널 출력
free_memory()              → VRAM 초기화
```

**터미널 출력 예시:**
```
────────────────────────────────────────────────────────────
  [통계] frog_01 — 총 5회 시도
    수치실패: 3 (60.0%)
      - ghosting: 2 (40.0%)
      - too_slow: 2 (40.0%)
    AI실패: 1 (20.0%)
    AI이슈 발생: 3회 (60.0%)
      - unnatural_movement: 3 (60.0%)
      - character_inconsistency: 2 (40.0%)
    성공: 1 (20.0%)
    보완 조치: (총 4회)
      - ai_adjust: 2 (50.0%)
      - seed_retry: 1 (25.0%)
      - action_switch: 1 (25.0%)
────────────────────────────────────────────────────────────
```

**파일 기록 예시 (validation_stats.txt):**
```
[2026-03-10 12:34:56] IMAGE: frog_01
  attempt 1: metric_fail  | too_slow, ghosting               [AI: unnatural_movement, character_inconsistency] → ai_adjust (pos:身体平滑运动) [VRAM:8133MB]
  attempt 2: metric_fail  | ghosting                         [AI: unnatural_movement, speed_too_slow] → ai_adjust (fps:14→16) [VRAM:8145MB]
  attempt 3: metric_fail  | too_slow, ghosting               [AI: unnatural_movement, character_inconsistency] → action_switch (→ 头部点头) [VRAM:8160MB]
  attempt 4: metric_fail  | no_motion, too_slow              → seed_retry (프롬프트 유지, seed만 교체) [VRAM:8155MB]
  attempt 5: metric_fail  | no_motion, too_slow              → seed_retry (프롬프트 유지, seed만 교체) [VRAM:8158MB]
  ---
```

---

### 8-7. pingpong / bg_type / bg_remove AI 동적 판단
**날짜:** 2026-03-10

**배경:**
기존에는 `pingpong=True`가 하드코딩되어 모든 영상이 왕복 루프였음.
오토바이 주행, 걸어가는 뒷모습 등 한 방향 이동 이미지에서는 역재생이 부자연스러움.

**구현:**

`VisionAnalysisResult`에 3개 필드 추가:
- `pingpong` (bool): 왕복(True) vs 단방향(False)
- `bg_type` (str): "solid"(단색) vs "scene"(장면)
- `bg_remove` (bool): 배경 제거 여부

Vision Analyzer 프롬프트에 **STEP 8** 추가:

**pingpong 판단 — CORE PRINCIPLE:**
"이 장면을 역재생하면 자연스러운가?" + "대상이 한 방향으로 이동하고 있는가?"
**포즈 기반 판단** — 배경 색상과 무관하게 포즈 자체로 이동 감지:
- 걷는 자세(다리 벌어짐), 뒷모습, 탈것 위 포즈 → pingpong=false
- 정지 자세(서 있음, 앉아 있음) → pingpong=true

**bg_type에 따른 BG 프롬프트 분기:**
| bg_type | positive | negative |
|---|---|---|
| solid | 纯白色背景 (순백 배경 유지) | 배경 변색/변암 금지 |
| scene | 背景保持不变 (배경 원본 유지) | 배경 소실/흐림/왜곡 금지 |

**검증 결과:**
| 이미지 | pingpong | bg_type | bg_remove |
|---|---|---|---|
| 개구리 (정면, 흰 배경) | True ✅ | solid ✅ | True ✅ |
| 오토바이 (도로 배경) | False ✅ | scene ✅ | False ✅ |
| 걸어가는 뒷모습 (흰 배경) | False ✅ | solid ✅ | True ✅ |

---

## 9. 3~4주차 개발 히스토리

> 3주차: 2026-03-11 ~ 2026-03-13 | 4주차: 2026-03-13 ~ 2026-03-16

---

### 9-1. Stage 1 모드 분류기 (wan_mode_classifier.py)
**날짜:** 2026-03-13

#### 개요
이미지 입력 시 WAN 모션 생성이 필요한지 사전 판단하는 Stage 1 게이트.
KEYFRAME_ONLY(강체) → WAN 건너뜀, MOTION_NEEDED(변형 가능) → WAN 생성 진행.

#### 판단 구조
```
PRE-CHECK:
  A) 이산적 주체 없음 → KEYFRAME_ONLY/pop
  B) 복합 주체 → 가장 큰 주체 기준
  C) 비정형(불/연기) → KEYFRAME_ONLY/wobble
  D) 장면 배경(road/sky/trees) → 강제 MOTION_NEEDED

Q1. 변형 가능 부위가 있는가?
  NO  → KEYFRAME_ONLY (suggested_action 포함)
  YES → MOTION_NEEDED
```

#### suggested_action (10가지)
nudge_horizontal, nudge_vertical, wobble, spin, bounce, pop, launch, float, parabolic, hop

#### 설계 원칙
- 프롬프트에 특정 오브젝트 이름 0건 (물리적 속성 기반 판단만)
- fallback: MOTION_NEEDED (안전 방향)
- temperature=0.1

---

### 9-2. Stage 2 모션 후 분류기 (wan_post_motion_classifier.py)
**날짜:** 2026-03-13, 2026-03-16 확장

#### 개요
WAN 생성 영상을 Gemini Vision으로 분석하여 CSS 키프레임 보강 필요 여부 판단.

#### 3가지 카테고리 (확장)
| 카테고리 | 유형 | 설명 | suggested_keyframe |
|---|---|---|---|
| A. TRAVEL | travel_lateral/vertical/diagonal | 프레임 밖으로 이탈 | launch, nudge_* |
| B. AMPLIFY | amplify_hop | 전신 점프 후 착지 | hop, bounce |
| | amplify_sway | 좌우 몸통 기울임 | wobble |
| | amplify_float | 느린 상하 부유 | float |
| C. NO_TRAVEL | no_travel | 부위만 움직임 | — |

#### amplify 카테고리 추가 배경
개구리 점프 영상이 "원위치 복귀 → no_travel"로 오판정되는 문제 해결.
핵심 구분: "전신이 올라가면 amplify_hop, 부위만 움직이면 no_travel"

#### 원본 이미지 선택적 처리
외부 생성 영상(grok 등)은 원본 이미지가 없을 수 있음.
원본 이미지 없이 영상만으로 분석 가능하도록 수정.

#### JSON 파싱 안정화
- max_output_tokens 500→2000 확대
- 불완전 JSON 복구 (잘린 따옴표/중괄호 닫기)
- 키워드 기반 추출 fallback

---

### 9-3. 키프레임 생성기 (wan_keyframe_generator.py)
**날짜:** 2026-03-13, 2026-03-16

#### 개요
KEYFRAME_ONLY 판정 이미지 또는 Stage 2 amplify 감지 시 CSS 키프레임 자동 생성.
물리 수식 기반 — 오브젝트 종류 하드코딩 없음.

#### 10가지 패턴
| 패턴 | 동작 | 수식 | loop | 기본 duration |
|---|---|---|---|---|
| nudge_horizontal | 좌우 감쇠 진동 | A·sin(ωt)·e^(-λt) | ✅ | 1200ms |
| nudge_vertical | 상하 감쇠 진동 | 동일 | ✅ | 1000ms |
| wobble | 좌우 회전 흔들림 | 감쇠 정현파 | ✅ | 1500ms |
| spin | Y축 회전 (scaleX) | scaleX 진동 | ✅ | 1000ms |
| bounce | squash & stretch | 키프레임 직접 | ✅ | 800ms |
| pop | 미세 스케일 펄스 | 키프레임 직접 | ✅ | 500ms |
| launch | 한 방향 가속 → 멈춤 | ease-out 곡선 | ❌ | 2000ms |
| float | 상하 부유 + 좌우 | sin/cos | ✅ | 3000ms |
| parabolic | 포물선 궤적 | y=-4h·t(t-1) | ❌ | 1500ms |
| hop | 제자리 올라갔다 내려오기 | y=-4h·t(t-1) | ❌ | 1200ms |

---

### 9-4. Lottie 변환기 (wan_lottie_converter.py)
**날짜:** 2026-03-13

#### 개요
투명 PNG 시퀀스를 Lottie JSON 애니메이션으로 변환. 래스터 내장(base64) 방식.

#### 프리셋별 용량
| 프리셋 | 해상도 | 최대 프레임 | 예상 용량 |
|---|---|---|---|
| original (기본) | 원본 | 전체 | ~7MB |
| web | 240px | 30 | ~2.1MB |
| web_hd | 360px | 40 | ~4.3MB |
| mobile | 180px | 24 | ~0.3MB |

---

### 9-5. 대시보드 프론트엔드 (wan_server.py + wan_dashboard.html)
**날짜:** 2026-03-13 ~ 2026-03-16

#### 개요
5단계 워크플로우를 시각적으로 테스트하는 독립 대시보드.
기존 파이프라인 코드 무수정 — wan_server.py + wan_dashboard.html 2개만 추가.

#### 기능 목록
- **파일 브라우저**: 대화상자로 서버 파일 시스템 탐색, 이미지 미리보기
- **Stage 1 분류**: 결과 표시, KEYFRAME_ONLY 시 키프레임 편집 UI 자동 표시
- **모션 생성**: 생성 횟수 입력(1~20), 실시간 영상 확인, 모델 선택 드롭다운
- **모델 선택**: WAN 2.1 Q3~Q5 / WAN 2.2 MoE Q3~Q4 / WAN 2.2 TI2V 5B
- **영상 탭**: "현재 이미지" / "전체 영상" 탭 분리
- **배경 제거 + Lottie**: 변환 후 lottie-web으로 미리보기
- **Stage 2 키프레임 판단**: amplify 감지 시 키프레임 자동 생성
- **키프레임 스테이지 뷰**: Lottie + CSS 키프레임 동기화 미리보기
  - 스테이지 크기 설정 (400×300 ~ 1024×768)
  - 배경 선택 (체크무늬/흰색/하늘/어두운/커스텀 이미지)
  - 오브젝트 크기 조절
  - 모션 모드: 점프(hop) / 부유(float) / 한 방향(oneway) / 진동(oscillate)
  - 슬라이더: 움직임 크기, 속도, 시작위치 X/Y, rotate, scale
  - "▶ 재생" 버튼 (1회 재생 후 원위치 복귀)
  - JSON 내보내기
- **실패 통계 그래프**: Chart.js (attempt별 막대 + 이슈 분포 도넛)

#### REST API 엔드포인트
| 엔드포인트 | 역할 |
|---|---|
| POST /api/classify | Stage 1 분류 |
| POST /api/generate | WAN 모션 생성 (비동기) |
| GET /api/status/<job_id> | 생성 진행 + 실시간 영상 목록 |
| GET /api/videos/<stem> | 특정 이미지 영상 목록 |
| GET /api/videos_all | 전체 영상 목록 |
| POST /api/select_video | 배경 제거 + Lottie 변환 |
| POST /api/classify_motion | Stage 2 판단 |
| POST /api/generate_keyframe | 키프레임 생성 |
| GET /api/stats/<stem> | 이미지별 생성 통계 |
| GET /api/stats_all | 전체 누적 통계 |
| GET /api/browse | 파일 브라우저 |
| GET /api/video_thumb/<path> | 영상 첫 프레임 썸네일 |
| GET /api/files/<path> | 정적 파일 서빙 |

---

### 9-6. 출력 폴더 분리
**날짜:** 2026-03-16

```
outputs/wan/
  ├── motion/              ← WAN 모션 생성 결과 (mp4, 배경제거, Lottie)
  ├── keyframe_only/       ← 키프레임만 (JSON, WAN 미사용)
  ├── motion_keyframe/     ← 모션 + 키프레임 (향후 사용)
  └── validation_stats.txt ← 검증 통계
```

---

### 9-7. Lottie + CSS 키프레임 동기화
**날짜:** 2026-03-16

#### 개요
WAN이 생성한 Lottie 모션(내부 애니메이션)과 CSS 키프레임(위치 이동)을 동기화하여 미리보기.

#### 구조
```
[outer div] ← CSS keyframe animate (translateY hop)
  └── [inner div] ← Lottie bodymovin (SVG 내부 애니메이션)
```

#### 동기화 방식
- Stage 2에서 hop 감지 → Lottie duration_ms 자동 전달 → 키프레임 duration 동기화
- "움직임 크기" 슬라이더로 점프 높이 실시간 조절
- "▶ 재생" 버튼으로 1회 재생 후 원위치 복귀

#### 발생했던 문제와 해결
- Lottie SVG transform과 CSS animate transform 충돌 → 2중 div 구조로 분리
- Web Animation API의 animate 누적 → getAnimations().cancel() 또는 전체 재생성
- setTimeout(50ms)으로 DOM paint 대기 후 animate 적용

---

## 10. PENDING

### 완료된 항목 (3~4주차)
- ✅ Stage 1 모드 분류기 (KEYFRAME_ONLY vs MOTION_NEEDED)
- ✅ Stage 2 모션 후 분류기 (TRAVEL / AMPLIFY / NO_TRAVEL)
- ✅ 키프레임 생성기 10종 (물리 수식 기반)
- ✅ Lottie 변환기 (4 프리셋)
- ✅ 대시보드 프론트엔드 (wan_server.py + wan_dashboard.html)
- ✅ 출력 폴더 분리 (motion / keyframe_only / motion_keyframe)
- ✅ 모델 선택 드롭다운 (WAN 2.1/2.2 양자화별)
- ✅ 영상 탭 분리 (현재 이미지 / 전체 영상)
- ✅ 키프레임 스테이지 뷰 + Lottie 동기화
- ✅ 실패 통계 그래프 (Chart.js)
- ✅ Stage 2 amplify 감지 (점프 모션 → hop 키프레임 자동 생성)
- ✅ 1회 재생 후 원위치 복귀 (hop)

### 남은 항목
- 🟡 키프레임 생성기 확장: 원운동(circular), 진자운동(pendulum), 제자리 회전(rotate_continuous), 떨림(vibrate) 추가
- 🟡 비전 분석 기반 모션 유형 자동 선택 → 프론트엔드 동적 옵션창 연동
- 🟡 MOTION + TRAVEL 동기화: Stage 2에서 TRAVEL 감지 시 Lottie + translate 키프레임 동기화
- 🟡 모델 성능 비교 기능: 생성 시간, VRAM, 성공률, 모션 스코어 등 모델별 비교
- 🟡 LLM 비전 모델 교체 테스트: Gemini 2.5-flash vs 2.5-pro, OpenAI/Claude 비교
- 🟢 `--cache-none` 옵션 테스트 (VRAM 누수 추가 방지)
- 🟢 characters2/ + vehicles/ 40개 이미지 일괄 실행
- 🟢 validation_stats.txt 기반 검증 임계값 최적화

### 파이프라인 실행 방법 (4주차 기준)
```bash
# 터미널 1 — ComfyUI
cd ~/ComfyUI && source venv/bin/activate
python main.py --listen

# 터미널 2 — WAN 대시보드 서버
cd ~/anim_pipeline && source animVenv/bin/activate
PYTHONPATH=~/anim_pipeline python image_pipeline/sprite_gen/wan_server.py
# → http://localhost:5001

# CLI 직접 실행 (선택)
cd ~/anim_pipeline && source animVenv/bin/activate
python3 -c "
import os, logging
logging.basicConfig(level=logging.INFO)
from dotenv import load_dotenv; load_dotenv()
from image_pipeline.sprite_gen.wan_backend import WanBackend
backend = WanBackend(
    api_key=os.environ['GEMINI_API_KEY'],
    comfyui_url='http://127.0.0.1:8188',
    output_dir='outputs/wan',
)
backend.generate('/path/to/image.png')
"
```

### 파일 현황 (4주차 기준, 13개)
| 파일 | 줄 수 | 역할 | 상태 |
|---|---|---|---|
| wan_backend.py | 2515 | 메인 오케스트레이터 | 수정 |
| wan_mode_classifier.py | 387 | Stage 1 분류 | 신규 |
| wan_post_motion_classifier.py | 406 | Stage 2 분류 | 신규 |
| wan_keyframe_generator.py | 513 | 키프레임 생성 (10종) | 신규 |
| wan_lottie_converter.py | 248 | Lottie JSON 변환 | 신규 |
| wan_server.py | 550+ | REST API 서버 | 신규 |
| wan_dashboard.html | 1300+ | 대시보드 프론트엔드 | 신규 |
| wan_vision_analyzer.py | 491 | 이미지 분석 | 미수정 |
| wan_validator.py | 708 | 수치 검증 | 미수정 |
| wan_ai_validator.py | 541 | AI 검증 | 미수정 |
| wan_mask_generator.py | 104 | 마스크 생성 | 미수정 |
| wan_bg_remover.py | 254 | 배경 제거 | 미수정 |

### 전체 처리 흐름 (4주차 기준)
```
이미지 입력
  │
  ├─ Stage 1: WanModeClassifier
  │    ├─ KEYFRAME_ONLY → WanKeyframeGenerator → JSON → 프론트 편집 → 완료
  │    └─ MOTION_NEEDED → WAN 생성 진행 ↓
  │
  ├─ WAN 모션 생성 (motion/ 폴더)
  │    ├─ Vision 분석 → 프롬프트 생성
  │    ├─ ComfyUI 생성 루프 (MAX_RETRIES회, 모델 선택 가능)
  │    ├─ 수치 검증 + AI 검증
  │    └─ 후처리 합성
  │
  ├─ 영상 선택 (수동, 현재이미지/전체 탭)
  │
  ├─ 배경 제거 → PNG 시퀀스 → Lottie JSON
  │
  └─ Stage 2: WanPostMotionClassifier
       ├─ NO_TRAVEL → 모션만 사용
       ├─ AMPLIFY → 키프레임 자동 생성 + 편집 UI (Lottie 동기화)
       └─ TRAVEL → 이동 키프레임 추가 (motion_keyframe/)
```

---
