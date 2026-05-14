# ============================================================
# VideoReTalking — Windows Setup Script
# Tested on: Windows 10/11, Python 3.8, CUDA 11.8 / 12.1
# ============================================================

param(
    [string]$EnvPath = "C:\Users\$env:USERNAME\conda_envs\video_retalking",
    [string]$CudaVersion = "cu118"   # Options: cu118, cu121, cpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "`n=== VideoReTalking Windows Setup ===" -ForegroundColor Cyan
Write-Host "Env path  : $EnvPath"
Write-Host "CUDA build: $CudaVersion`n"

# ── 1. Create conda environment ─────────────────────────────
Write-Host "[1/6] Creating conda environment (Python 3.8)..." -ForegroundColor Yellow
conda create --prefix "$EnvPath" python=3.8 -y
if ($LASTEXITCODE -ne 0) { throw "conda create failed" }

# ── 2. Activate ─────────────────────────────────────────────
Write-Host "[2/6] Activating environment..." -ForegroundColor Yellow
conda activate "$EnvPath"

# ── 3. Core conda packages (pre-built binaries, no compiler) ─
Write-Host "[3/6] Installing conda packages (ffmpeg, lmdb, dlib)..." -ForegroundColor Yellow
conda install -c conda-forge ffmpeg lmdb dlib -y
if ($LASTEXITCODE -ne 0) { throw "conda install failed" }

# ── 4. PyTorch ───────────────────────────────────────────────
Write-Host "[4/6] Installing PyTorch ($CudaVersion)..." -ForegroundColor Yellow
switch ($CudaVersion) {
    "cu118" { pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 }
    "cu121" { pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 }
    "cpu"   { pip install torch torchvision }
    default { throw "Unknown CudaVersion '$CudaVersion'. Use cu118, cu121, or cpu." }
}
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed" }

# ── 5. Project requirements ──────────────────────────────────
Write-Host "[5/6] Installing project requirements..." -ForegroundColor Yellow
# Install basicsr without deps first (lmdb already installed via conda)
pip install basicsr==1.4.2 --no-deps
pip install addict future pyyaml requests yapf
pip install -r requirements.txt --no-deps 2>$null
# Full install for remaining packages
pip install kornia==0.5.1 face-alignment==1.3.4 ninja==1.10.2.3 einops==0.4.1 `
            facexlib==0.2.5 librosa==0.9.2 "gradio>=3.7.0" `
            "numpy>=1.23.4,<2.0" scipy scikit-image opencv-python Pillow tqdm
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# ── 6. Environment variable ──────────────────────────────────
Write-Host "[6/6] Setting KMP_DUPLICATE_LIB_OK for this session..." -ForegroundColor Yellow
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

Write-Host "`n=== Setup complete! ===" -ForegroundColor Green
Write-Host "To run inference:" -ForegroundColor Cyan
Write-Host '  $env:KMP_DUPLICATE_LIB_OK = "TRUE"'
Write-Host '  python inference.py --face examples/face/women.mp4 --audio examples/audio/women_audio.wav --outfile results/output.mp4'
