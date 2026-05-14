---
title: VideoReTalking
emoji: 🎭
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: true
python_version: "3.10"
---

[![HF Spaces Demo](https://img.shields.io/badge/Hugging%20Face-Spaces%20Demo-blue)](https://huggingface.co/spaces/Vedant3907/video-retalking)

# VideoReTalking — Audio-based Lip Synchronization for Talking Head Videos

[![ArXiv](https://img.shields.io/badge/ArXiv-2211.14758-red)](https://arxiv.org/abs/2211.14758)
[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://vinthony.github.io/video-retalking/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinthony/video-retalking/blob/main/quick_demo.ipynb)

> **SIGGRAPH Asia 2022** — Kun Cheng, Xiaodong Cun, Yong Zhang et al.

Given a talking-head video and a new audio track, VideoReTalking produces a lip-synced output with matching expressions — no manual alignment required.

---

## How It Works

The pipeline has three sequential stages:

1. **Expression normalization** — rewrites every frame to a canonical neutral expression using DNet
2. **Audio-driven lip sync** — drives the mouth with the target audio using LNet
3. **Face enhancement** — sharpens and harmonizes the final face with GPEN/GFPGAN

![Pipeline](docs/static/images/pipeline.png)

---

## Setup (Windows)

### Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU with CUDA 11.8+ (or CPU — slower)
- [Git](https://git-scm.com/)

### Option A — Automated (PowerShell)

```powershell
git clone https://github.com/your-username/video-retalking.git
cd video-retalking

# For CUDA 11.8 (default):
.\setup_windows.ps1

# For CUDA 12.1:
.\setup_windows.ps1 -CudaVersion cu121

# CPU only:
.\setup_windows.ps1 -CudaVersion cpu
```

### Option B — Manual Step-by-Step

```powershell
# 1. Create environment
conda create --prefix "C:\conda_envs\video_retalking" python=3.8 -y
conda activate "C:\conda_envs\video_retalking"

# 2. Install packages that need pre-built binaries (no C++ compiler required)
conda install -c conda-forge ffmpeg lmdb dlib -y

# 3. Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install basicsr (skip lmdb since it's already installed via conda)
pip install basicsr==1.4.2 --no-deps
pip install addict future pyyaml requests yapf

# 5. Install remaining requirements
pip install kornia==0.5.1 face-alignment==1.3.4 einops==0.4.1 facexlib==0.2.5 ^
            librosa==0.9.2 "numpy>=1.23.4,<2.0" scipy scikit-image opencv-python Pillow tqdm
```

### Download Pretrained Models

Download checkpoints from [Google Drive](https://drive.google.com/drive/folders/18rhjMpxK8LVVxf7PI6XwOidt8Vouv_H0?usp=share_link) and place them in `./checkpoints/`:

```
checkpoints/
├── DNet.pt
├── LNet.pth
├── ENet.pth
├── GFPGANv1.3.pth
└── ...
```

---

## Running Inference

```powershell
# Required once per terminal session
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

python inference.py `
  --face examples/face/women.mp4 `
  --audio examples/audio/women_audio.wav `
  --outfile results/output.mp4
```

### Optional Parameters

| Flag | Default | Description |
|---|---|---|
| `--exp_img` | `neutral` | Expression template: `neutral`, `smile`, or path to an image |
| `--up_face` | — | Upper face expression: `surprise` or `angry` |
| `--LNet_batch_size` | `8` | Batch size for lip-sync network |
| `--img_size` | `384` | Processing resolution |

---

## Compatibility Notes

This repo has been patched to work with modern Python packages (2024+). The original codebase targeted older versions that are no longer available. Patches applied:

| File | Issue | Fix |
|---|---|---|
| `utils/audio.py` | `librosa.filters.mel()` positional args removed | Changed to keyword args |
| `third_part/face3d/extract_kp_videos.py` | `face_alignment.LandmarksType._2D` renamed | Use `TWO_D` with fallback |
| `utils/inference_utils.py` | Same as above | Same fix |
| `utils/alignment_stit.py` | Same as above | Same fix |
| `third_part/face3d/extract_kp_videos.py` | `keypoints` unbound when no face detected | Initialize before loop |
| `third_part/face3d/util/preprocess.py` | `np.VisibleDeprecationWarning` removed in NumPy 2.x | Guard with `hasattr` |
| `third_part/face_detection/utils.py` | `np.int` removed in NumPy 1.24 | Changed to `np.int32` |
| `basicsr/data/degradations.py` (site-packages) | `torchvision.transforms.functional_tensor` removed | Changed to `functional` |
| `inference.py` | Dual OpenMP runtime crash on Windows | Set `KMP_DUPLICATE_LIB_OK=TRUE` |

---

## Project Structure

```
video-retalking/
├── inference.py              # Main inference entry point
├── checkpoints/              # Pretrained model weights
├── examples/
│   ├── face/                 # Sample face videos
│   └── audio/                # Sample audio files
├── models/                   # DNet, LNet, ENet definitions
├── utils/                    # Audio processing, face alignment, blending
├── third_part/               # GPEN, GFPGAN, face3d, face_detection
├── results/                  # Output videos
├── requirements.txt          # Python dependencies
└── setup_windows.ps1         # One-click Windows setup script
```

---

## Citation

```bibtex
@misc{cheng2022videoretalking,
  title   = {VideoReTalking: Audio-based Lip Synchronization for Talking Head Video Editing In the Wild},
  author  = {Kun Cheng and Xiaodong Cun and Yong Zhang and Menghan Xia and Fei Yin
             and Mingrui Zhu and Xuan Wang and Jue Wang and Nannan Wang},
  year    = {2022},
  eprint  = {2211.14758},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

---

## Acknowledgements

Built on top of:
[Wav2Lip](https://github.com/Rudrabha/Wav2Lip) •
[PIRenderer](https://github.com/RenYurui/PIRender) •
[GFP-GAN](https://github.com/TencentARC/GFPGAN) •
[GPEN](https://github.com/yangxy/GPEN) •
[ganimation_replicate](https://github.com/donydchen/ganimation_replicate) •
[STIT](https://github.com/rotemtzaban/STIT)
