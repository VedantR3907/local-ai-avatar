# Local AI Avatar — Real-Time Lip-Sync Voice Agent

[![GitHub](https://img.shields.io/badge/GitHub-local--ai--avatar-black?logo=github)](https://github.com/VedantR3907/local-ai-avatar)
[![ArXiv](https://img.shields.io/badge/ArXiv-2211.14758-red)](https://arxiv.org/abs/2211.14758)

Speak → AI replies → Watch the avatar say it with synced lips.

<video src="https://github.com/user-attachments/assets/b1977f39-8649-41ac-b68a-9a9b2177ec1c" controls autoplay loop muted width="100%"></video>

A fully hands-free voice pipeline built on top of VideoReTalking:

**Mic** → **Silero VAD** (auto-detects speech) → **Whisper STT** → **Ollama Cloud LLM** → **gTTS** → **VideoReTalking lip-sync** → **Video plays automatically**

No button presses. No Enter key. Just talk.

---

## Demo

| Input | Output |
|-------|--------|
| `examples/face/women.mp4` + `examples/audio/women_audio.wav` | Lip-synced talking head video |

> Generated videos are in `outputs/` (not tracked in git — run it yourself!)

---

## How It Works

### Voice Agent Pipeline (`voice_agent_simple.py`)

```
You speak
   └─► Silero VAD detects speech start/end automatically
         └─► Whisper transcribes your words
               └─► Ollama Cloud LLM generates a reply
                     └─► gTTS converts reply to speech
                           └─► VideoReTalking renders lip-synced video
                                 └─► Video opens automatically
```

### VideoReTalking Core Pipeline

1. **Expression normalization** — rewrites every frame to a neutral expression (DNet)
2. **Audio-driven lip sync** — drives the mouth region with target audio (LNet)
3. **Face enhancement** — sharpens and harmonizes the final face (GPEN/GFPGAN)

---

## Quick Setup (Windows)

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU with CUDA 11.8+ (strongly recommended — CPU is very slow)
- [Git](https://git-scm.com/)
- [FFmpeg](https://ffmpeg.org/) (or install via `conda install -c conda-forge ffmpeg`)

### Step 1 — Clone

```powershell
git clone https://github.com/VedantR3907/local-ai-avatar.git
cd local-ai-avatar
```

### Step 2 — Install Dependencies

**Option A — Automated (one command):**

```powershell
.\setup_windows.ps1
```

**Option B — Manual:**

```powershell
# Create conda environment
conda create --prefix "C:\conda_envs\video_retalking" python=3.8 -y
conda activate "C:\conda_envs\video_retalking"

# Install binaries that need pre-built wheels
conda install -c conda-forge ffmpeg lmdb dlib -y

# PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install basicsr==1.4.2 --no-deps
pip install addict future pyyaml requests yapf
pip install kornia==0.5.1 face-alignment==1.3.4 einops==0.4.1 facexlib==0.2.5
pip install librosa==0.9.2 "numpy>=1.23.4,<2.0" scipy scikit-image opencv-python Pillow tqdm

# Voice agent dependencies
pip install openai-whisper gtts sounddevice pyaudio silero-vad ollama
```

### Step 3 — Download Model Checkpoints

The model weights (~4 GB) are hosted on Hugging Face:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Vedant3907/local-ai-avatar-checkpoints",
                  local_dir="checkpoints")
```

Or manually download from the [original Google Drive](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and place files in `./checkpoints/`.

Expected files:

```
checkpoints/
├── DNet.pt                          (172 MB)
├── LNet.pth                        (1463 MB)
├── ENet.pth                         (546 MB)
├── GFPGANv1.3.pth                   (332 MB)
├── GPEN-BFR-512.pth                 (270 MB)
├── face3d_pretrain_epoch_20.pth     (275 MB)
├── RetinaFace-R50.pth               (104 MB)
├── ParseNet-latest.pth               (81 MB)
├── shape_predictor_68_face_landmarks.dat (95 MB)
├── wav2lip_gan.pth                  (415 MB)
├── 30_net_gen.pth                    (32 MB)
└── expression.mat
```

---

## Running

### Hands-Free Voice Agent

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:TORCHDYNAMO_DISABLE="1"
& "C:\conda_envs\video_retalking\python.exe" voice_agent_simple.py
```

Just speak. It listens automatically, detects when you stop, then runs the full pipeline.
The output video opens automatically when ready (~2-5 min on a mid-range GPU).

**Config (top of `voice_agent_simple.py`):**

```python
OLLAMA_API_KEY = "your-key-here"   # https://ollama.com/settings/keys
OLLAMA_MODEL   = "gpt-oss:120b"    # or any model on Ollama Cloud
FACE_VIDEO     = "examples/face/women.mp4"  # swap with any face video
```

### Inference Only (no voice agent)

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
$env:TORCHDYNAMO_DISABLE="1"

python inference.py `
  --face   examples/face/women.mp4 `
  --audio  examples/audio/women_audio.wav `
  --outfile results/output.mp4 `
  --LNet_batch_size 4 `
  --fast
```

> `--fast` skips face enhancement for ~3x faster rendering.  
> `--LNet_batch_size 4` reduces VRAM usage (safe for 6–8 GB GPUs).

---

## Compatibility Patches Applied

This repo patches the original VideoReTalking to work with modern packages (2024+):

| File | Issue | Fix |
|------|-------|-----|
| `utils/audio.py` | `librosa.filters.mel()` positional args removed | Keyword args |
| `third_part/face3d/extract_kp_videos.py` | `LandmarksType._2D` renamed to `TWO_D` | Fallback lookup |
| `utils/alignment_stit.py` | `PIL.Image.ANTIALIAS` removed in Pillow 10 | Use `LANCZOS` |
| `utils/ffhq_preprocess.py` | Same Pillow issue | Same fix |
| `third_part/face3d/util/preprocess.py` | NumPy inhomogeneous array error | Explicit float cast |
| `third_part/face_detection/utils.py` | `np.int` removed in NumPy 1.24 | Use `np.int32` |
| `inference.py` | Dual OpenMP crash on Windows | `KMP_DUPLICATE_LIB_OK=TRUE` |
| `inference.py` | Last batch CUDA recompilation (5–20 min slowdown) | Pad remainder batch |

---

## Project Structure

```
local-ai-avatar/
├── voice_agent_simple.py     # Hands-free voice agent (main feature)
├── inference.py              # Core lip-sync inference
├── examples/
│   ├── face/women.mp4        # Sample face video
│   └── audio/women_audio.wav # Sample audio
├── models/                   # DNet, LNet, ENet model definitions
├── utils/                    # Audio, face alignment, blending utilities
├── third_part/               # GPEN, GFPGAN, face3d, face_detection
├── checkpoints/              # Model weights (download separately)
├── setup_windows.ps1         # One-click Windows setup
└── requirements.txt          # Python dependencies
```

---

## Built On

[VideoReTalking (SIGGRAPH Asia 2022)](https://github.com/OpenTalker/video-retalking) •
[Whisper](https://github.com/openai/whisper) •
[Silero VAD](https://github.com/snakers4/silero-vad) •
[Ollama](https://ollama.com) •
[gTTS](https://github.com/pndurette/gTTS) •
[GFPGAN](https://github.com/TencentARC/GFPGAN) •
[GPEN](https://github.com/yangxy/GPEN)
