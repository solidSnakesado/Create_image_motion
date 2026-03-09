# WAN I2V 파이프라인 — WSL 완전 새 환경 셋업 가이드

> 대상: Windows WSL2에서 아무것도 설치되지 않은 상태에서 시작
> 레포: https://github.com/solidSnakesado/Create_image_motion/tree/dev
> 최종 목표: 스프라이트 PNG를 넣으면 모션 APNG가 자동 생성되는 파이프라인 실행

---

## 전체 흐름 요약

```
[1] WSL2 + Ubuntu 설치
    ↓
[2] NVIDIA 드라이버 + CUDA 설치
    ↓
[3] 시스템 패키지 설치 (python3, ffmpeg, git)
    ↓
[4] 프로젝트 코드 클론 (anim_pipeline)
    ↓
[5] Python 가상환경 + 의존성 설치
    ↓
[6] ComfyUI 설치 (별도 디렉토리)
    ↓
[7] ComfyUI 커스텀 노드 설치
    ↓
[8] WAN 모델 4종 다운로드
    ↓
[9] 환경변수 설정 (.env)
    ↓
[10] 실행 확인
```

---

## 1. WSL2 + Ubuntu 설치

Windows PowerShell (관리자 권한)에서 실행:

```powershell
wsl --install -d Ubuntu-24.04
```

설치 후 Ubuntu 터미널이 열리면 사용자 이름과 비밀번호를 설정합니다.
이후 모든 작업은 Ubuntu 터미널에서 진행합니다.

WSL2 버전 확인:

```bash
wsl --version
# WSL 버전: 2.x.x 이상이어야 GPU 패스스루 지원
```

---

## 2. NVIDIA 드라이버 + CUDA 설치

### 2-1. Windows 측 NVIDIA 드라이버

WSL2에서 GPU를 사용하려면 **Windows 측에** 최신 NVIDIA 드라이버가 설치되어 있어야 합니다.
WSL2 내부에는 별도로 드라이버를 설치하지 않습니다.

Windows에서 확인:
- NVIDIA 공식 사이트에서 최신 Game Ready / Studio 드라이버 설치
- 또는 GeForce Experience로 업데이트

WSL2 내부에서 GPU 인식 확인:

```bash
nvidia-smi
```

GPU 정보가 표시되면 정상입니다. 표시되지 않으면 Windows 드라이버를 업데이트하세요.

### 2-2. CUDA Toolkit (WSL2 내부)

```bash
# CUDA 키링 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit 설치
sudo apt install -y cuda-toolkit-12-6

# 환경변수 추가
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version
```

> 주의: WSL2에서는 `wsl-ubuntu` 저장소를 사용합니다 (`ubuntu2404`가 아님).

---

## 3. 시스템 패키지 설치

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    build-essential \
    nano
```

각 패키지의 역할:

| 패키지 | 용도 |
|---|---|
| python3, pip, venv | Python 실행 + 가상환경 |
| ffmpeg | 영상 프레임 추출, 재인코딩, APNG 생성 |
| git | 코드 클론 + 버전 관리 |
| build-essential | C 확장 빌드 (numpy 등) |
| nano | 텍스트 편집기 (.env 수정용) |

---

## 4. 프로젝트 코드 클론

```bash
cd ~
git clone https://github.com/solidSnakesado/Create_image_motion.git anim_pipeline
cd anim_pipeline
git checkout dev
```

클론 후 구조 확인:

```bash
ls image_pipeline/sprite_gen/
```

아래 파일들이 보이면 정상:

```
wan_backend.py
wan_vision_analyzer.py
wan_validator.py
wan_ai_validator.py
wan_mask_generator.py
wan_bg_remover.py
__init__.py
```

---

## 5. Python 가상환경 + 의존성 설치

```bash
cd ~/anim_pipeline

# 가상환경 생성
python3 -m venv animVenv

# 가상환경 활성화
source animVenv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 의존성 설치
pip install -r requirements.txt
```

`requirements.txt`에 포함된 패키지:

| 패키지 | 용도 |
|---|---|
| numpy | 수치 계산 (검증 메트릭) |
| Pillow | 이미지 처리 (전처리, 마스크, APNG) |
| scipy | flood fill 배경 제거 (ndimage.label) |
| python-dotenv | .env 파일에서 환경변수 로드 |
| google-genai | Gemini Vision API (이미지 분석 + AI 검증) |

---

## 6. ComfyUI 설치

ComfyUI는 WAN 모델을 실행하는 **별도 서버**입니다. 프로젝트 코드와 다른 디렉토리에 설치합니다.

```bash
# 기존의 animVenv 가상이라면 deactivate
deactivate
cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# ComfyUI 전용 가상환경 (anim_pipeline과 분리)
python3 -m venv venv
source venv/bin/activate

# PyTorch 설치 (CUDA 12.4 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ComfyUI 의존성 설치
pip install -r requirements.txt
```

> 가상환경을 분리하는 이유: ComfyUI는 PyTorch + CUDA 전체 스택이 필요하지만,
> anim_pipeline은 PyTorch 없이 HTTP 통신만 하므로 의존성 충돌을 방지합니다.

---

## 7. ComfyUI 커스텀 노드 설치

ComfyUI venv가 활성화된 상태에서 진행합니다.

```bash
# ComfyUI venv 활성화 확인
cd ~/ComfyUI
source venv/bin/activate

cd custom_nodes

# GGUF 로더 — WAN 양자화 모델(.gguf) 로드용
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF && pip install -r requirements.txt && cd ..

# Video Helper Suite — 영상 출력(mp4) 생성용
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ..
```

---

## 8. WAN 모델 다운로드

4개 모델 파일을 ComfyUI의 models 디렉토리에 배치합니다.

### 8-1. HuggingFace CLI 설치

```bash
# ComfyUI venv 활성화 상태에서
pip install huggingface-hub
```

### 8-2. 모델 다운로드

```bash
cd ~/ComfyUI/models

# (1) WAN I2V UNet — 영상 생성 본체 (~8GB)
mkdir -p unet
huggingface-cli download city96/Wan2.1-I2V-14B-480P-gguf \
    wan2.1-i2v-14b-480p-Q3_K_S.gguf \
    --local-dir unet/

# (2) CLIP Vision — 이미지 인코더 (~1.2GB)
mkdir -p clip_vision
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/clip_vision/clip_vision_h.safetensors \
    --local-dir clip_vision/

# (3) CLIP 텍스트 인코더 (~16GB)
mkdir -p clip
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
    --local-dir clip/

# (4) VAE — 디코더 (~300MB)
mkdir -p vae
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
    split_files/vae/wan_2.1_vae.safetensors \
    --local-dir vae/
```

> 주의: HuggingFace에서 다운로드 시 파일이 하위 폴더에 저장될 수 있습니다.
> 다운로드 후 모델 파일이 정확한 위치에 있는지 확인하세요.

### 8-3. 모델 파일 위치 확인

```bash
ls -lh ~/ComfyUI/models/unet/wan2.1-i2v-14b-480p-Q3_K_S.gguf
ls -lh ~/ComfyUI/models/clip_vision/clip_vision_h.safetensors
ls -lh ~/ComfyUI/models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ls -lh ~/ComfyUI/models/vae/wan_2.1_vae.safetensors
```

4개 파일이 모두 존재하면 정상입니다. 파일이 하위 폴더에 들어간 경우 올바른 위치로 이동시킵니다.

---

## 9. 환경변수 설정

### 9-1. Gemini API 키 발급

1. https://aistudio.google.com/ 접속
2. 좌측 메뉴 → API keys → Create API key
3. 생성된 키 복사

### 9-2. .env 파일 생성

```bash
cd ~/anim_pipeline
cp .env.example .env
nano .env
```

아래 내용으로 수정:

```env
GEMINI_API_KEY=여기에_발급받은_키_붙여넣기
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_ROOT=~/ComfyUI
LLM_PROVIDER=none
```

`Ctrl+O` → `Enter` → `Ctrl+X`로 저장 후 나옵니다.

### 9-3. Git 사용자 설정 (최초 1회)

```bash
git config --global user.email "your-email@example.com"
git config --global user.name "your-github-username"
```

---

## 10. 실행 확인

두 개의 터미널(또는 tmux 세션)이 필요합니다.

### 터미널 1 — ComfyUI 서버 시작

```bash
cd ~/ComfyUI
source venv/bin/activate
python main.py --listen
```

정상 실행 시 아래 메시지가 출력됩니다:

```
Starting server
To see the GUI go to: http://0.0.0.0:8188
```

### 터미널 2 — WAN 파이프라인 실행

```bash
cd ~/anim_pipeline
source animVenv/bin/activate

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
backend.generate('/path/to/your/image.png')
"
```

`/path/to/your/image.png`을 실제 이미지 경로로 변경합니다.

### 실행 결과물

생성 성공 시 `outputs/wan/` 디렉토리에 아래 파일들이 생성됩니다:

```
outputs/wan/
├── image_name_processed.png              # 480×480 전처리 이미지
├── image_name_attempt01.mp4              # 1차 시도 영상
├── image_name_attempt02.mp4              # 2차 시도 영상 (실패 시)
├── ...
└── image_name_transparent/
    ├── image_name_attempt01_frame_0000.png   # 투명 PNG 시퀀스
    ├── image_name_attempt01_frame_0001.png
    ├── ...
    └── image_name_attempt01_transparent.apng # 최종 APNG
```

---

## 트러블슈팅

### nvidia-smi 가 작동하지 않음

Windows 측 NVIDIA 드라이버가 최신이 아니거나, WSL2가 아닌 WSL1을 사용 중일 수 있습니다.

```powershell
# PowerShell에서 WSL 버전 확인
wsl -l -v
# VERSION이 2인지 확인. 1이면:
wsl --set-version Ubuntu-24.04 2
```

### ModuleNotFoundError: No module named 'image_pipeline'

반드시 `~/anim_pipeline` 디렉토리에서 실행해야 합니다.

```bash
cd ~/anim_pipeline && source animVenv/bin/activate
```

### ComfyUI 시작 시 torch 관련 에러

PyTorch와 CUDA 버전이 맞지 않을 수 있습니다. ComfyUI venv에서 재설치:

```bash
cd ~/ComfyUI && source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### ComfyUI에서 모델을 찾지 못함

모델 파일이 정확한 경로에 있는지 확인합니다. HuggingFace 다운로드 시 `split_files/` 같은 하위 폴더가 생길 수 있습니다.

```bash
# 예: clip_vision이 하위 폴더에 있는 경우
mv ~/ComfyUI/models/clip_vision/split_files/clip_vision/clip_vision_h.safetensors \
   ~/ComfyUI/models/clip_vision/
```

### Gemini API 429 Rate Limit

Google AI Studio 무료 티어의 분당 요청 제한입니다. 잠시 후 재실행하거나 유료 플랜을 사용합니다.

### 생성은 되지만 APNG 배경이 회색

정상 동작입니다. WAN 모델이 배경을 회색(RGB 206~217)으로 생성하는 경우가 있으며, `wan_bg_remover.py`의 tolerance=50이 이를 커버하여 투명으로 제거합니다.

### VRAM 부족 (Out of Memory)

Q3_K_S 모델(~8GB)은 VRAM 10GB 이상을 권장합니다. 다른 GPU 프로세스를 종료하거나, ComfyUI 시작 시 메모리 관리 옵션을 추가합니다:

```bash
python main.py --listen --lowvram
```

---

## 하드웨어 요구사항 요약

| 항목 | 최소 | 권장 |
|---|---|---|
| OS | Windows 10 21H2+ (WSL2) | Windows 11 |
| GPU | NVIDIA, VRAM 10GB+ | VRAM 16GB+ |
| RAM | 16GB | 32GB |
| 디스크 | 40GB (모델 포함) | 60GB+ |
| Python | 3.10+ | 3.11~3.12 |

---

## 파이프라인 실행 흐름 요약

```
[입력: 스프라이트 PNG]
        │
        ▼
[전처리] 480×480 흰 캔버스 + White Anchor
        │
        ▼
[Gemini Vision 분석] 액션·프롬프트·수치 자동 결정
        │
        ▼
[ComfyUI WAN 2.1] 영상 생성 (GPU)
        │
        ▼
[수치 검증 8항목] no_motion / too_fast / ghosting 등
        │
        ▼
[AI 검증] Gemini가 영상을 직접 보고 품질 판정
        │
        ├─ 실패 → 파라미터 조정 후 재시도 (최대 10회)
        │         3회 연속 실패 → 다른 동작으로 전환
        │
        ▼ 통과
[후처리 합성] 고정 부위에 원본 이미지 덮어쓰기
        │
        ▼
[배경 제거] flood fill → 투명 배경
        │
        ▼
[출력: APNG] 루프 애니메이션 완성
```
