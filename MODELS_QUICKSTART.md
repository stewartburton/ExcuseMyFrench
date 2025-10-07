# Models Quick Download Guide

**Updated:** October 7, 2025
**Based on:** Verified working links as of today

This is a simplified quick-start guide. For detailed instructions, see `docs/MODEL_DOWNLOADS.md`.

---

## ✅ Working Download Links (Verified)

### 1. Wav2Lip Models (~500MB)

**Directory:** `models/wav2lip/`

| File | Size | Working Link | Status |
|------|------|--------------|--------|
| s3fd.pth | 86MB | [Download](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) | ✅ Working |
| Wav2Lip-SD-GAN.pt | 140MB | [HuggingFace](https://huggingface.co/spaces/CVPR/Wav2Lip/blob/main/Wav2Lip-SD-GAN.pt) | ✅ Working |
| Wav2Lip-SD-NOGAN.pt | 140MB | [HuggingFace](https://huggingface.co/spaces/CVPR/Wav2Lip/blob/main/Wav2Lip-SD-NOGAN.pt) | ✅ Working |

**Note:** Actual filenames are `Wav2Lip-SD-*.pt`, NOT `wav2lip*.pth` as mentioned in older documentation.

**Instructions:**
1. Create directory: `mkdir -p models/wav2lip`
2. Download s3fd.pth from link above (rename from `s3fd-619a316812.pth` to `s3fd.pth`)
3. Visit HuggingFace links above, click download button (↓) for both Wav2Lip-SD-*.pt files
4. Save all files to `models/wav2lip/`

**Alternative:** Visit [Wav2Lip GitHub](https://github.com/Rudrabha/Wav2Lip) for Google Drive links (older naming: `wav2lip.pth`, `wav2lip_gan.pth`)

---

### 2. SadTalker Models (~2GB)

**Directory:** `models/sadtalker/checkpoints/` and `models/sadtalker/gfpgan/weights/`

#### ✅ V0.0.2 Models (Newer, Recommended)

| File | Size | Repository | Status |
|------|------|------------|--------|
| SadTalker_V0.0.2_256.safetensors | 553MB | vinthony/SadTalker-V002rc | ✅ Working |
| SadTalker_V0.0.2_512.safetensors | 553MB | vinthony/SadTalker-V002rc | ✅ Working |

**Download Links:**
- https://huggingface.co/vinthony/SadTalker-V002rc/tree/main
- Click each filename → Click download button (↓)
- Save to: `models/sadtalker/checkpoints/`

#### ✅ Main Models (Older, Alternative)

| File | Size | Repository | Status |
|------|------|------------|--------|
| auido2exp_00300-model.pth | 343MB | vinthony/SadTalker | ✅ Working |
| auido2pose_00140-model.pth | 343MB | vinthony/SadTalker | ✅ Working |

**Download Links:**
- https://huggingface.co/vinthony/SadTalker/tree/main
- Click each filename → Click download button (↓)
- Save to: `models/sadtalker/checkpoints/`

#### ✅ GFPGAN Weights (Enhancement)

| File | Size | Source | Status |
|------|------|--------|--------|
| alignment_WFLW_4HG.pth | 59MB | [facexlib v0.1.0](https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth) | ✅ Working |
| headpose_hopenet.pth | 13MB | [facexlib v0.2.2](https://github.com/xinntao/facexlib/releases/download/v0.2.2/headpose_hopenet.pth) | ✅ Working |
| GFPGANv1.4.pth | 348MB | [GFPGAN v1.3.0](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) | ✅ Working |

**Instructions:**
1. Create directory: `mkdir -p models/sadtalker/gfpgan/weights`
2. Download each file from the links above
3. Save to: `models/sadtalker/gfpgan/weights/`

---

### 3. Stable Diffusion 2.1 Base (~4GB)

**Auto-downloads during training** - no manual download needed!

Just run training:
```bash
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
```

The model will download automatically to your HuggingFace cache.

**Manual download (optional):**
1. Visit: https://huggingface.co/stabilityai/stable-diffusion-2-1-base
2. Accept license
3. Use: `huggingface-cli download stabilityai/stable-diffusion-2-1-base`

---

## ❌ Known Non-Working Links

These links from older documentation are **NOT WORKING** as of October 2025:

- ❌ OneDrive/SharePoint links for Wav2Lip (404 errors)
- ❌ vinthony/SadTalker `checkpoints/` folder on main branch (doesn't exist)
- ❌ Old HuggingFace paths with `mapping_00109` and `mapping_00229` files

---

## 🚀 Quick Command Line Downloads

### Using wget (Linux/Mac/WSL):

```bash
# Create directories
mkdir -p models/wav2lip
mkdir -p models/sadtalker/checkpoints
mkdir -p models/sadtalker/gfpgan/weights

# Wav2Lip face detection
cd models/wav2lip
wget https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -O s3fd.pth
cd ../..

# SadTalker V0.0.2 (recommended)
cd models/sadtalker/checkpoints
wget https://huggingface.co/vinthony/SadTalker-V002rc/resolve/main/SadTalker_V0.0.2_256.safetensors
wget https://huggingface.co/vinthony/SadTalker-V002rc/resolve/main/SadTalker_V0.0.2_512.safetensors
cd ../../..

# GFPGAN weights
cd models/sadtalker/gfpgan/weights
wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth
wget https://github.com/xinntao/facexlib/releases/download/v0.2.2/headpose_hopenet.pth
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
cd ../../../..
```

### Using HuggingFace CLI:

```bash
# Install if needed
pip install huggingface-hub

# Download SadTalker V0.0.2
huggingface-cli download vinthony/SadTalker-V002rc \
  SadTalker_V0.0.2_256.safetensors \
  SadTalker_V0.0.2_512.safetensors \
  --local-dir models/sadtalker/checkpoints
```

---

## ✓ Verification

After downloading, verify with:

```bash
# Check Wav2Lip
ls -lh models/wav2lip/
# Expected: s3fd.pth (~86MB), Wav2Lip-SD-GAN.pt (~140MB), Wav2Lip-SD-NOGAN.pt (~140MB)

# Check SadTalker
ls -lh models/sadtalker/checkpoints/
# Expected: SadTalker_V0.0.2_256.safetensors (~553MB), SadTalker_V0.0.2_512.safetensors (~553MB)

# Check GFPGAN
ls -lh models/sadtalker/gfpgan/weights/
# Expected: 3 files (~420MB total)
```

Or run the verification script:
```bash
python scripts/verify_models.py
```

---

## 📋 Minimum Required Models

**To get started quickly, download only these:**

1. ✅ **Wav2Lip:**
   - `s3fd.pth` (86MB) - Required for face detection
   - `Wav2Lip-SD-GAN.pt` (140MB) - High quality lip-sync
   - `Wav2Lip-SD-NOGAN.pt` (140MB) - Faster lip-sync (alternative)

2. ✅ **SadTalker (choose one set):**
   - V0.0.2 models (2 files, ~1.1GB) - **Recommended**
   - OR main models (2 files, ~686MB) - Alternative

3. ✅ **GFPGAN:**
   - All 3 files (~420MB) - Required for face enhancement

4. ✅ **Stable Diffusion:**
   - Auto-downloads during training

**Total download size:** ~2-2.5GB minimum

---

## 🆘 Troubleshooting

### Download fails with 404 error
- ✅ Use the links in this guide (verified October 2025)
- ❌ Don't use old OneDrive/SharePoint links from 2023-2024

### HuggingFace download stalls
- Use browser download instead of CLI
- Try download manager (JDownloader, Free Download Manager)
- Check disk space

### Can't find Wav2Lip models
- Visit GitHub: https://github.com/Rudrabha/Wav2Lip
- Look for Google Drive link in "Getting the weights" section
- **Note:** Older docs mention `wav2lip.pth` and `wav2lip_gan.pth`, but newer versions use `Wav2Lip-SD-*.pt`
- You only need ONE of the two Wav2Lip-SD models (GAN for quality, NOGAN for speed)

---

## 📚 Full Documentation

For complete details, troubleshooting, and alternative methods:
- **Full guide:** `docs/MODEL_DOWNLOADS.md`
- **Troubleshooting:** `docs/TROUBLESHOOTING.md`
- **Testing:** `docs/TESTING.md`

---

**Last Verified:** October 7, 2025
**Next Review:** Monthly or after community reports broken links
