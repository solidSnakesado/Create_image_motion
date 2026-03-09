# WAN I2V 파이프라인 — WSL 완전 새 환경 셋업 가이드

> 대상: Windows WSL2에서 아무것도 설치되지 않은 상태에서 시작
> 레포: https://github.com/solidSnakesado/Create_image_motion/tree/dev
> 최종 목표: 스프라이트 PNG를 넣으면 모션 APNG가 자동 생성되는 파이프라인 실행
> 최종 검증일: 2026-03-09 (RTX 5070, Ubuntu 24.04, CUDA 12.6, Python 3.12)

---

## 전체 흐름 요약

```
[1] WSL2 + Ubuntu 설치
    ↓
[2] NVIDIA 드라이버 확인 + CUDA Toolkit 설치
    ↓
[3] 시스템 패키지 설치 (python3, ffmpeg, git)
    ↓
[4] 프로젝트 코드 클론 (anim_pipeline)
    ↓
[5] Python 가상환경 + 의존성 설치
    ↓
[6] ComfyUI 설치 (별도 디렉토리, 별도 가상환경)
    ↓
[7] ComfyUI 커스텀 노드 설치
    ↓
[8] WAN 모델 4종 다운로드 + 경로 정리
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

WSL 버전 확인 (PowerShell에서):

```powershell
wsl -l -v
```

VERSION이 **2**인지 확인합니다. 1이면:

```powershell
wsl --set-version Ubuntu-24.04 2
```

---

## 2. NVIDIA 드라이버 확인 + CUDA Toolkit 설치

### 2-1. Windows 측 NVIDIA 드라이버

WSL2에서 GPU를 사용하려면 **Windows 측에** 최신 NVIDIA 드라이버가 설치되어 있어야 합니다.
WSL2 내부에는 별도로 드라이버를 설치하지 않습니다.

Windows PowerShell에서 확인:

```powershell
nvidia-smi
```

드라이버 버전이 **470 이상**이어야 WSL2 GPU 패스스루를 지원합니다.

### 2-2. WSL2 내부에서 GPU 인식 확인

Ubuntu 터미널에서:

```bash
nvidia-smi
```

> ⚠️ "GPU access blocked by the operating system" 에러 시:
> PowerShell에서 `wsl --shutdown` 실행 후 Ubuntu를 다시 열면 해결됩니다.

### 2-3. CUDA Toolkit 설치 (WSL2 내부)

```bash
# CUDA 키링 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA Toolkit 설치
# ⚠️ cuda-toolkit-12-4는 Ubuntu 24.04에서 libtinfo5 의존성 문제로 설치 실패
# cuda-toolkit-12-6 사용
sudo apt install -y cuda-toolkit-12-6

# 환경변수 추가
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version
```

> 주의: WSL2에서는 `wsl-ubuntu` 저장소를 사용합니다 (`ubuntu2404`가 아님).

> ⚠️ `dpkg: error: dpkg frontend lock was locked by another process` 에러 시:
> `unattended-upgrades` 자동 업데이트가 실행 중입니다. 잠시 기다리거나:
> ```bash
> sudo lsof /var/lib/dpkg/lock-frontend  # PID 확인
> sudo kill <PID>                         # 프로세스 종료 후 재시도
> ```

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

설치 확인:

```bash
ffmpeg -version | head -1
python3 --version
```

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
wan_backend.py  wan_vision_analyzer.py  wan_validator.py
wan_ai_validator.py  wan_mask_generator.py  wan_bg_remover.py
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

# pip 업그레이드 + 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```

설치 확인:

```bash
pip list | grep -E "google-genai|numpy|Pillow|scipy"
```

4개 패키지가 모두 보이면 정상입니다.

---

## 6. ComfyUI 설치

ComfyUI는 WAN 모델을 실행하는 **별도 서버**입니다. 반드시 anim_pipeline과 **다른 가상환경**에 설치합니다.

> 가상환경을 분리하는 이유: ComfyUI는 PyTorch + CUDA 전체 스택이 필요하지만,
> anim_pipeline은 PyTorch 없이 HTTP 통신만 하므로 의존성 충돌을 방지합니다.

```bash
# animVenv가 활성화되어 있으면 먼저 비활성화
deactivate

cd ~
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# ComfyUI 전용 가상환경
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# PyTorch 설치
# ⚠️ GPU 아키텍처에 따라 설치 명령이 다릅니다
#
# RTX 50XX 시리즈 (Blackwell, sm_120) → cu128 nightly 권장
#   stable(2.10.0)은 sm_120 최적화가 부족하여 생성 속도가 ~50% 느림
#   nightly(2.12.0.dev+)는 Blackwell 최적화 포함 → 정상 속도
# RTX 40XX 시리즈 (Ada Lovelace, sm_89) → cu126 stable 이상
# RTX 30XX 시리즈 (Ampere, sm_86) → cu124 stable 이상
#
# 확인 방법: nvidia-smi로 GPU 이름 확인

# RTX 50XX (Blackwell) — cu128 nightly (권장, 최적화 포함)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 50XX (Blackwell) — cu128 stable (동작하지만 ~50% 느림)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# RTX 40XX 이하 — cu126 stable
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 설치 후 GPU 호환성 확인
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# True와 GPU 이름이 출력되면 정상
# RTX 50XX는 버전이 2.12.0.dev 이상이어야 최적 성능

# ComfyUI 의존성 설치
pip install -r requirements.txt
```

---

## 7. ComfyUI 커스텀 노드 설치

ComfyUI venv가 활성화된 상태에서 진행합니다.

```bash
cd ~/ComfyUI/custom_nodes

# GGUF 로더 — WAN 양자화 모델(.gguf) 로드용
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ComfyUI-GGUF && pip install -r requirements.txt && cd ..

# Video Helper Suite — 영상 출력(mp4) 생성용
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
cd ComfyUI-VideoHelperSuite && pip install -r requirements.txt && cd ..
```

---

## 8. WAN 모델 다운로드 + 경로 정리

4개 모델 파일을 ComfyUI의 models 디렉토리에 다운로드합니다.
ComfyUI venv가 활성화된 상태에서 진행합니다.

> ⚠️ `huggingface-cli` 명령이 작동하지 않을 수 있습니다 (huggingface-hub 버전에 따라 다름).
> Python 코드로 직접 다운로드하는 방식을 사용합니다.

```bash
cd ~/ComfyUI/models
```

### 8-1. 모델 다운로드 (하나씩 순서대로 실행)

```bash
# (1) WAN I2V UNet — 영상 생성 본체 (~8GB, 약 7분)
mkdir -p unet
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('city96/Wan2.1-I2V-14B-480P-gguf', 'wan2.1-i2v-14b-480p-Q3_K_S.gguf', local_dir='unet/')
"

# (2) CLIP Vision — 이미지 인코더 (~1.2GB, 약 1분)
mkdir -p clip_vision
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/clip_vision/clip_vision_h.safetensors', local_dir='clip_vision/')
"

# (3) CLIP 텍스트 인코더 (~6.3GB, 약 6분)
mkdir -p clip
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors', local_dir='clip/')
"

# (4) VAE — 디코더 (~243MB, 약 15초)
mkdir -p vae
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('Comfy-Org/Wan_2.1_ComfyUI_repackaged', 'split_files/vae/wan_2.1_vae.safetensors', local_dir='vae/')
"
```

### 8-2. 파일 경로 정리

HuggingFace에서 다운로드하면 (2)(3)(4)가 `split_files/` 하위 폴더에 저장됩니다.
ComfyUI가 모델을 찾을 수 있도록 올바른 위치로 이동합니다.

```bash
# 파일 이동
mv clip_vision/split_files/clip_vision/clip_vision_h.safetensors clip_vision/
mv clip/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors clip/
mv vae/split_files/vae/wan_2.1_vae.safetensors vae/

# 빈 폴더 정리
rm -rf clip_vision/split_files clip/split_files vae/split_files
```

### 8-3. 최종 경로 확인

```bash
ls -lh unet/wan2.1-i2v-14b-480p-Q3_K_S.gguf
ls -lh clip_vision/clip_vision_h.safetensors
ls -lh clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors
ls -lh vae/wan_2.1_vae.safetensors
```

4개 파일이 모두 각 폴더 바로 아래에 있으면 정상입니다.

| 파일 | 크기 | 경로 |
|---|---|---|
| WAN UNet | ~7.4GB | `models/unet/wan2.1-i2v-14b-480p-Q3_K_S.gguf` |
| CLIP Vision | ~1.2GB | `models/clip_vision/clip_vision_h.safetensors` |
| CLIP Text | ~6.3GB | `models/clip/umt5_xxl_fp8_e4m3fn_scaled.safetensors` |
| VAE | ~243MB | `models/vae/wan_2.1_vae.safetensors` |

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

# push 시 토큰 재입력 방지
git config --global credential.helper store
```

### 9-4. 워크플로우 파일 배치

레포에서 최신 코드를 받아 워크플로우 파일을 ComfyUI에 배치합니다.

```bash
cd ~/anim_pipeline
git pull origin dev

# 워크플로우 파일을 ComfyUI 경로에 복사
mkdir -p ~/ComfyUI/user/default/workflows/
cp workflows/wan21_native_i2v_1.json ~/ComfyUI/user/default/workflows/
```

> 워크플로우 파일이 없어도 `wan_backend.py`의 코드 빌드 fallback으로 동작합니다.
> 단, 검증된 미세 설정(cfg, sampler 등)을 재현하려면 파일 배치를 권장합니다.

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
    ├── ...
    └── image_name_attempt01_transparent.apng # 최종 APNG
```

---

## 트러블슈팅

### nvidia-smi: "GPU access blocked by the operating system"

WSL을 완전 재시작하면 해결됩니다.

```powershell
# Windows PowerShell에서
wsl --shutdown
```

이후 Ubuntu를 다시 열고 `nvidia-smi` 확인.

### nvidia-smi가 아예 없음

Windows 측 NVIDIA 드라이버가 설치되지 않았거나, WSL1을 사용 중일 수 있습니다.

```powershell
# PowerShell에서 WSL 버전 확인
wsl -l -v
# VERSION이 2인지 확인
```

### cuda-toolkit-12-4 설치 실패 (libtinfo5)

Ubuntu 24.04에서 `libtinfo5` 의존성 문제가 발생합니다. **cuda-toolkit-12-6**을 설치합니다.

```bash
sudo apt install -y cuda-toolkit-12-6
```

### dpkg lock 에러

`unattended-upgrades` 자동 업데이트가 실행 중입니다.

```bash
sudo lsof /var/lib/dpkg/lock-frontend  # PID 확인
sudo kill <PID>                         # 종료 후 재시도
```

### ModuleNotFoundError: No module named 'image_pipeline'

반드시 `~/anim_pipeline` 디렉토리에서 실행해야 합니다.

```bash
cd ~/anim_pipeline && source animVenv/bin/activate
```

### huggingface-cli: command not found

`huggingface-hub` 버전에 따라 CLI가 PATH에 등록되지 않을 수 있습니다.
Python 코드로 직접 다운로드합니다 (8단계 참조).

### ComfyUI에서 모델을 찾지 못함

HuggingFace 다운로드 시 `split_files/` 하위 폴더가 생성됩니다.
파일을 올바른 위치로 이동해야 합니다 (8-2단계 참조).

### ComfyUI 시작 시 torch 관련 에러

PyTorch와 GPU 아키텍처가 맞지 않을 수 있습니다.

**증상 1 — "CUDA capability sm_120 is not compatible":**
RTX 50XX (Blackwell) GPU에 cu126 이하 PyTorch를 설치한 경우. cu128 이상이 필요합니다.

```
NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible
current PyTorch install supports CUDA capabilities sm_50 ~ sm_90
Please install PyTorch with CUDA: 12.8 13.0
```

**증상 2 — "no kernel image is available for execution on the device":**
위 호환성 문제로 CUDA 커널이 실행되지 않는 경우. ComfyUI가 2~4초 만에 완료되고 비디오 파일이 생성되지 않습니다.

**해결:**

```bash
cd ~/ComfyUI && source venv/bin/activate
pip uninstall torch torchvision torchaudio -y

# RTX 50XX → cu128 nightly (권장)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# RTX 40XX 이하 → cu126
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 호환성 확인
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### ComfyUI 생성 속도가 비정상적으로 느림 (RTX 50XX)

PyTorch stable(2.10.0+cu128)을 사용하면 Blackwell 최적화가 부족하여 생성 속도가 약 50% 느려집니다.
ComfyUI 시작 시 아래 경고가 표시됩니다:

```
WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.
backend triton: disabled: True
backend cuda: disabled: True
```

**해결:** PyTorch nightly를 설치합니다.

```bash
cd ~/ComfyUI && source venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 버전 확인 — 2.12.0.dev 이상이어야 최적 성능
python -c "import torch; print(torch.__version__)"
```

### Gemini API 429 Rate Limit

Google AI Studio 무료 티어의 분당 요청 제한입니다.
잠시 후 재실행하거나 유료 플랜을 사용합니다.

### 배경이 회색으로 생성됨

정상 동작입니다. WAN 모델이 배경을 회색(RGB 206~217)으로 생성하는 경우가 있으며,
`wan_bg_remover.py`의 tolerance=50이 이를 커버하여 투명으로 제거합니다.

### VRAM 부족 (Out of Memory)

Q3_K_S 모델(~8GB)은 VRAM 10GB 이상을 권장합니다.
다른 GPU 프로세스를 종료하거나 ComfyUI 시작 시:

```bash
python main.py --listen --lowvram
```

---

## 하드웨어 요구사항 요약

| 항목 | 최소 | 권장 |
|---|---|---|
| OS | Windows 10 21H2+ (WSL2) | Windows 11 |
| GPU | NVIDIA, VRAM 10GB+ | VRAM 12GB+ |
| RAM | 16GB | 32GB |
| 디스크 | 40GB (모델 포함) | 60GB+ |
| Python | 3.10+ | 3.12 |
| CUDA Toolkit | 12.6 | 12.6 |

---

## 검증 완료 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop (12GB VRAM, Blackwell sm_120) |
| OS | Ubuntu 24.04 on WSL2 (WSL 2.6.3) |
| Windows 드라이버 | 595.71 |
| CUDA Toolkit | 12.6 (V12.6.85) |
| Python | 3.12.3 |
| PyTorch | 2.12.0.dev+cu128 (nightly, Blackwell 최적화) |
| ComfyUI | 0.16.4 (2026-03-09) |

> ⚠️ RTX 50XX는 CUDA capability sm_120으로, PyTorch cu126 이하는 sm_90까지만 지원합니다.
> 반드시 cu128 이상을 설치해야 하며, **nightly 버전(2.12.0.dev+)**을 권장합니다.
> stable(2.10.0+cu128)은 동작하지만 최적화 부족으로 생성 속도가 약 50% 느립니다.

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
