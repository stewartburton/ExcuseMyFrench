# Model Downloads Guide

**Last Updated:** October 7, 2025

This guide provides manual download instructions for all AI models required by ExcuseMyFrench. Automated downloads may fail due to file size, authentication, or network issues, so manual download is often more reliable.

---

## Table of Contents

1. [Wav2Lip Models](#wav2lip-models) (~500MB)
2. [SadTalker Models](#sadtalker-models) (~2GB)
3. [Wan 2.2 Models](#wan-22-models) (~10-28GB)
4. [Stable Diffusion Base Model](#stable-diffusion-base-model) (~4GB)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

---

## Wav2Lip Models

**Size:** ~500MB total
**Required for:** Lip-sync animation
**Directory:** `models/wav2lip/`

### Files Needed

1. **Wav2Lip-SD-NOGAN.pt** (~140MB) - Main model (faster)
2. **Wav2Lip-SD-GAN.pt** (~140MB) - GAN variant (higher quality)
3. **s3fd.pth** (~86MB) - Face detection model (required)

**Note:** You only need ONE of the two Wav2Lip models (GAN or NOGAN). Older versions used `wav2lip.pth` naming.

### Manual Download Steps

#### Option 1: Direct Browser Download (Recommended)

1. **Create directory:**
   ```bash
   mkdir -p models/wav2lip
   ```

2. **Download s3fd.pth (face detection):**
   - Visit: https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth
   - Right-click → Save As...
   - Save to: `models/wav2lip/s3fd.pth`
   - **Note:** File may auto-download with name `s3fd-619a316812.pth` - rename to `s3fd.pth`

3. **Download Wav2Lip models:**

   **Option A: HuggingFace (Recommended - SD models):**
   - Visit: https://huggingface.co/spaces/CVPR/Wav2Lip/tree/main
   - Download: `Wav2Lip-SD-GAN.pt` (140MB) - Higher quality
   - Download: `Wav2Lip-SD-NOGAN.pt` (140MB) - Faster processing
   - Click download button (↓) for each file
   - Save to: `models/wav2lip/`

   **Option B: Official GitHub (Google Drive - older versions):**
   - Visit: https://github.com/Rudrabha/Wav2Lip
   - Look for "Getting the weights" section in README
   - Files may be named `wav2lip.pth` and `wav2lip_gan.pth` (older naming)
   - Save to: `models/wav2lip/`

   **Note:** The SD (Stable Diffusion enhanced) versions are recommended. You only need ONE model (either GAN or NOGAN).

#### Option 2: Using wget (Linux/Mac)

```bash
cd models/wav2lip

# Face detection model
wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O s3fd.pth

# For the OneDrive links, you'll need to use browser download
# The direct links require SharePoint authentication
```

#### Option 3: Using PowerShell (Windows)

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path models\wav2lip

# Download face detection model
Invoke-WebRequest -Uri "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -OutFile "models\wav2lip\s3fd.pth"

# For wav2lip.pth and wav2lip_gan.pth, use browser download from OneDrive
```

### Alternative Sources

If official links are down, check:

- **HuggingFace Mirror:** https://huggingface.co/spaces/badayvedat/Wav2Lip
- **GitHub Releases:** https://github.com/Rudrabha/Wav2Lip/releases
- **Community Mirrors:** Search "wav2lip checkpoint download" (use caution, verify checksums)

### Verification

```bash
# Check files exist
ls -lh models/wav2lip/

# Expected output (SD models - recommended):
# s3fd.pth                (~86MB)
# Wav2Lip-SD-GAN.pt       (~140MB)
# Wav2Lip-SD-NOGAN.pt     (~140MB)

# OR (older versions from Google Drive):
# s3fd.pth                (~86MB)
# wav2lip.pth             (~193MB)
# wav2lip_gan.pth         (~193MB)
```

---

## SadTalker Models

**Size:** ~2GB total
**Required for:** High-quality facial animation
**Directory:** `models/sadtalker/`

### Files Needed

#### Main Checkpoints (models/sadtalker/checkpoints/)

**Note:** The V0.0.2 models are in a separate repository: `vinthony/SadTalker-V002rc`

**From vinthony/SadTalker (main models):**
1. **auido2exp_00300-model.pth** (~343MB) - Audio to expression mapping
2. **auido2pose_00140-model.pth** (~343MB) - Audio to pose mapping

**From vinthony/SadTalker-V002rc (newer version, recommended):**
3. **SadTalker_V0.0.2_256.safetensors** (~553MB)
4. **SadTalker_V0.0.2_512.safetensors** (~553MB)

#### GFPGAN Enhancement Weights (models/sadtalker/gfpgan/weights/)

1. **alignment_WFLW_4HG.pth** (~59MB)
2. **headpose_hopenet.pth** (~13MB)
3. **GFPGANv1.4.pth** (~348MB)

### Manual Download Steps

#### Step 1: Create Directories

```bash
mkdir -p models/sadtalker/checkpoints
mkdir -p models/sadtalker/gfpgan/weights
```

#### Step 2: Download Main Checkpoints

**Option A: HuggingFace Web Interface (Recommended)**

1. **For main models**, visit: https://huggingface.co/vinthony/SadTalker/tree/main
   - Download: `auido2exp_00300-model.pth`
   - Download: `auido2pose_00140-model.pth`
   - Click each filename → Click download button (↓) → Save to `models/sadtalker/checkpoints/`

2. **For V0.0.2 models** (newer, recommended), visit: https://huggingface.co/vinthony/SadTalker-V002rc/tree/main
   - Download: `SadTalker_V0.0.2_256.safetensors`
   - Download: `SadTalker_V0.0.2_512.safetensors`
   - Click each filename → Click download button (↓) → Save to `models/sadtalker/checkpoints/`

**Option B: HuggingFace CLI**

```bash
# Install HuggingFace CLI if not already installed
pip install huggingface-hub

# Download main checkpoint files
huggingface-cli download vinthony/SadTalker \
  auido2exp_00300-model.pth \
  auido2pose_00140-model.pth \
  --local-dir models/sadtalker/checkpoints

# Download V0.0.2 models (newer, recommended)
huggingface-cli download vinthony/SadTalker-V002rc \
  SadTalker_V0.0.2_256.safetensors \
  SadTalker_V0.0.2_512.safetensors \
  --local-dir models/sadtalker/checkpoints
```

**Option C: Direct URLs with wget**

```bash
cd models/sadtalker/checkpoints

# Main models
wget https://huggingface.co/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth
wget https://huggingface.co/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth

# V0.0.2 models (newer, recommended)
wget https://huggingface.co/vinthony/SadTalker-V002rc/resolve/main/SadTalker_V0.0.2_256.safetensors
wget https://huggingface.co/vinthony/SadTalker-V002rc/resolve/main/SadTalker_V0.0.2_512.safetensors

cd ../../..
```

#### Step 3: Download GFPGAN Weights

**Face Alignment Model:**

```bash
cd models/sadtalker/gfpgan/weights

# Browser download:
# Visit: https://github.com/xinntao/facexlib/releases/tag/v0.1.0
# Download: alignment_WFLW_4HG.pth

# OR wget:
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth
```

**Head Pose Model:**

```bash
# Browser download:
# Visit: https://github.com/xinntao/facexlib/releases/tag/v0.2.0
# Download: headpose_hopenet.pth

# OR wget:
wget https://github.com/xinntao/facexlib/releases/download/v0.2.0/headpose_hopenet.pth
```

**GFPGAN Enhancement:**

```bash
# Browser download:
# Visit: https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.0
# Download: GFPGANv1.4.pth

# OR wget:
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

### Verification

```bash
# Check main checkpoints
ls -lh models/sadtalker/checkpoints/

# Expected output (main models):
# auido2exp_00300-model.pth             (~343MB)
# auido2pose_00140-model.pth            (~343MB)

# OR (V0.0.2 models - newer, recommended):
# SadTalker_V0.0.2_256.safetensors      (~553MB)
# SadTalker_V0.0.2_512.safetensors      (~553MB)

# You can have both sets, or just the V0.0.2 models

# Check GFPGAN weights
ls -lh models/sadtalker/gfpgan/weights/

# Expected output:
# alignment_WFLW_4HG.pth                (~59MB)
# headpose_hopenet.pth                  (~13MB)
# GFPGANv1.4.pth                        (~348MB)
```

---

## Wan 2.2 Models

**Size:** 10-28GB (model dependent)
**Required for:** Text-to-video / Image-to-video generation (optional)
**Directory:** `models/wan2.2/`

### ⚠️ Important Notes

- **Status:** As of October 2025, Wan 2.2 may still be in development or limited release
- **VRAM Requirements:**
  - 5B model: 12GB+ VRAM (RTX 4070, RTX 3090, etc.)
  - 14B model: 24GB+ VRAM (RTX 4090, A6000, etc.)
- **Not Required:** This is optional for v1.0. You can use static images + animation (Wav2Lip/SadTalker) instead

### Research Official Sources First

1. **Check HuggingFace:**
   - Search: https://huggingface.co/models?search=wan
   - Look for official releases from AI research labs

2. **Check GitHub:**
   - Search: https://github.com/search?q=wan+2.2+video
   - Look for official repositories with model releases

3. **Check Research Papers:**
   - ArXiv: https://arxiv.org/search/?query=wan+2.2+video
   - Papers often include model download links

### Manual Download Steps (Once Available)

#### Option 1: HuggingFace CLI (If Available)

```bash
# Create directory
mkdir -p models/wan2.2

# Download 5B model (recommended for RTX 4070)
huggingface-cli download [OFFICIAL-REPO-NAME]/wan2.2-5B \
  --local-dir models/wan2.2/TI2V-5B \
  --local-dir-use-symlinks False

# OR download 14B model (requires 24GB+ VRAM)
huggingface-cli download [OFFICIAL-REPO-NAME]/wan2.2-14B \
  --local-dir models/wan2.2/TI2V-14B \
  --local-dir-use-symlinks False
```

#### Option 2: Git LFS Clone

```bash
# If model is hosted as a Git repository with LFS
git lfs install
git clone https://huggingface.co/[OFFICIAL-REPO]/wan2.2-5B models/wan2.2/TI2V-5B
```

#### Option 3: Manual Browser Download

1. Visit the official HuggingFace or GitHub repository
2. Navigate to the "Files and versions" tab
3. Download each checkpoint file individually
4. Place in: `models/wan2.2/TI2V-5B/` or `models/wan2.2/TI2V-14B/`

### Alternative: Use Stable Diffusion + Animation Instead

If Wan 2.2 is unavailable or too large:

1. Generate static images with Stable Diffusion (DreamBooth)
2. Animate with Wav2Lip or SadTalker
3. This approach is proven, faster, and uses less VRAM

### Verification

```bash
# Check model files
ls -lh models/wan2.2/TI2V-5B/

# Expected output (varies by model):
# config.json
# pytorch_model.bin or model.safetensors (~10GB)
# tokenizer files
# other config files
```

---

## Stable Diffusion Base Model

**Size:** ~4GB
**Required for:** DreamBooth training (custom character model)
**Directory:** HuggingFace cache (automatic)

### Manual Download Steps

#### Option 1: Auto-download During Training (Recommended)

```bash
# The model will auto-download on first training run
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
```

#### Option 2: Pre-download with HuggingFace CLI

```bash
# Login to HuggingFace (accept model license on website first)
huggingface-cli login

# Download Stable Diffusion 2.1 base model
huggingface-cli download stabilityai/stable-diffusion-2-1-base
```

#### Option 3: Accept License and Download via Web

1. Visit: https://huggingface.co/stabilityai/stable-diffusion-2-1-base
2. Read and accept the license agreement
3. Click "Files and versions"
4. Download manually or use HuggingFace CLI after login

### Verification

```bash
# Check HuggingFace cache
ls -lh ~/.cache/huggingface/hub/ | grep stable-diffusion

# OR on Windows:
dir %USERPROFILE%\.cache\huggingface\hub | findstr stable-diffusion
```

---

## Verification

### Automated Verification Script

Create and run `scripts/verify_models.py`:

```python
#!/usr/bin/env python3
"""Verify all downloaded models."""

import os
from pathlib import Path

def check_model(path, expected_size_mb=None):
    """Check if model exists and optionally verify size."""
    if not path.exists():
        print(f"❌ Missing: {path}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    status = "✅" if size_mb > 1 else "⚠️"

    print(f"{status} Found: {path.name} ({size_mb:.1f} MB)")

    if expected_size_mb and abs(size_mb - expected_size_mb) > 10:
        print(f"   ⚠️ Warning: Expected ~{expected_size_mb}MB, got {size_mb:.1f}MB")

    return True

# Check Wav2Lip
print("\n=== Wav2Lip Models ===")
check_model(Path("models/wav2lip/s3fd.pth"), 86)

# Check for SD models (newer, recommended)
sd_gan = Path("models/wav2lip/Wav2Lip-SD-GAN.pt")
sd_nogan = Path("models/wav2lip/Wav2Lip-SD-NOGAN.pt")

# Check for older models
old_gan = Path("models/wav2lip/wav2lip_gan.pth")
old_nogan = Path("models/wav2lip/wav2lip.pth")

if sd_gan.exists() or sd_nogan.exists():
    check_model(sd_gan, 140)
    check_model(sd_nogan, 140)
elif old_gan.exists() or old_nogan.exists():
    check_model(old_gan, 193)
    check_model(old_nogan, 193)
else:
    print("❌ No Wav2Lip model found (need either SD or older version)")

# Check SadTalker
print("\n=== SadTalker Models ===")
# Main models (older)
check_model(Path("models/sadtalker/checkpoints/auido2exp_00300-model.pth"), 343)
check_model(Path("models/sadtalker/checkpoints/auido2pose_00140-model.pth"), 343)
# V0.0.2 models (newer, recommended)
check_model(Path("models/sadtalker/checkpoints/SadTalker_V0.0.2_256.safetensors"), 553)
check_model(Path("models/sadtalker/checkpoints/SadTalker_V0.0.2_512.safetensors"), 553)
check_model(Path("models/sadtalker/gfpgan/weights/alignment_WFLW_4HG.pth"), 59)
check_model(Path("models/sadtalker/gfpgan/weights/headpose_hopenet.pth"), 13)
check_model(Path("models/sadtalker/gfpgan/weights/GFPGANv1.4.pth"), 348)

# Check Wan 2.2 (optional)
print("\n=== Wan 2.2 Models (Optional) ===")
wan_5b_dir = Path("models/wan2.2/TI2V-5B")
if wan_5b_dir.exists():
    print(f"✅ Wan 2.2 5B directory exists")
    for f in wan_5b_dir.glob("*.bin"):
        check_model(f)
    for f in wan_5b_dir.glob("*.safetensors"):
        check_model(f)
else:
    print("⚠️ Wan 2.2 not downloaded (optional)")

print("\n=== Summary ===")
print("Run: python scripts/verify_models.py")
```

Run verification:

```bash
python scripts/verify_models.py
```

---

## Troubleshooting

### Issue: Download Speed Too Slow

**Solutions:**
- Use a download manager (JDownloader, Free Download Manager)
- Try downloading during off-peak hours
- Use a VPN if location is throttled
- Split large files using `aria2c` or `axel` (supports resume)

### Issue: OneDrive/SharePoint Links Not Working

**Solutions:**
- Open link in an incognito/private browser window
- Clear browser cache and cookies
- Try a different browser (Chrome, Firefox, Edge)
- Check if you need to sign in to Microsoft account
- Use the GitHub releases page instead

### Issue: HuggingFace Download Fails

**Solutions:**
- Login with `huggingface-cli login`
- Accept model license on the website first
- Use `--resume-download` flag for interrupted downloads
- Try the web interface download instead

### Issue: File Corrupted After Download

**Solutions:**
- Re-download the file
- Check file size matches expected size
- Verify checksum if provided
- Try different download method (CLI vs browser)

### Issue: Not Enough Disk Space

**Solutions:**
- Free up space (delete temporary files, old downloads)
- Use external hard drive for models directory
- Download only essential models first (skip GAN variants)
- Consider using smaller Wan 2.2 model (5B vs 14B)

### Issue: "Permission Denied" When Saving

**Solutions:**
- Run terminal/PowerShell as administrator
- Check folder permissions
- Make sure directory exists: `mkdir -p models/xxx`
- Try saving to a different location first, then move

---

## Quick Reference: Download Checklist

Use this checklist to track your downloads:

### Required Models

- [ ] **Wav2Lip - Face Detection** (s3fd.pth, ~86MB)
- [ ] **Wav2Lip - Main Model** (wav2lip.pth, ~193MB)
- [ ] **Stable Diffusion 2.1** (auto-downloads during training)

### Optional but Recommended

- [ ] **Wav2Lip - GAN Model** (wav2lip_gan.pth, ~193MB) - Higher quality
- [ ] **SadTalker - All Checkpoints** (~2GB total) - Best quality animation
- [ ] **SadTalker - GFPGAN Weights** (~420MB total) - Face enhancement

### Optional Advanced

- [ ] **Wan 2.2 - 5B or 14B** (~10-28GB) - Text/Image-to-video generation

---

## Resources

### Official Repositories

- **Wav2Lip:** https://github.com/Rudrabha/Wav2Lip
- **SadTalker:** https://github.com/OpenTalker/SadTalker
- **Stable Diffusion:** https://github.com/Stability-AI/stablediffusion
- **GFPGAN:** https://github.com/TencentARC/GFPGAN

### HuggingFace Models

- **SadTalker:** https://huggingface.co/vinthony/SadTalker
- **Stable Diffusion 2.1:** https://huggingface.co/stabilityai/stable-diffusion-2-1-base

### Community Resources

- **HuggingFace Discussions:** https://huggingface.co/spaces
- **GitHub Issues:** Check each repository's issues for download help
- **Reddit:** r/StableDiffusion, r/MachineLearning

---

## Next Steps After Download

1. **Verify Downloads:**
   ```bash
   python scripts/verify_models.py
   ```

2. **Test Models:**
   ```bash
   # Test environment
   python scripts/demo.py

   # Test DreamBooth training
   python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
   ```

3. **Run Pipeline:**
   ```bash
   # Full pipeline test
   make run-pipeline
   ```

4. **Read Documentation:**
   - `QUICKSTART.md` - Getting started guide
   - `docs/TESTING.md` - Testing procedures
   - `docs/TROUBLESHOOTING.md` - Common issues

---

**Last Updated:** October 7, 2025
**Maintained By:** ExcuseMyFrench Project

For issues or questions, see: `docs/TROUBLESHOOTING.md` or open a GitHub issue.
