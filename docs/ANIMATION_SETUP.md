# Animation Setup Guide

This guide covers setup for lip-sync animation tools (SadTalker, Wav2Lip) and video generation models (Wan 2.2) for the Excuse My French pipeline.

## Table of Contents

- [Overview](#overview)
- [SadTalker Setup](#sadtalker-setup)
- [Wav2Lip Setup](#wav2lip-setup)
- [Wan 2.2 Setup](#wan-22-setup)
- [Testing Animation](#testing-animation)

---

## Overview

The Excuse My French pipeline supports multiple animation methods:

1. **SadTalker** (Recommended) - State-of-the-art lip-sync with head pose
2. **Wav2Lip** (Alternative) - Lightweight lip-sync model
3. **Wan 2.2** (Optional) - Advanced video generation from text/images

---

## SadTalker Setup

### What is SadTalker?

SadTalker generates realistic talking head videos from a single image and audio. It produces natural head movements and facial expressions along with accurate lip-sync.

### System Requirements

- **GPU**: NVIDIA GPU with 6GB+ VRAM (recommended)
- **CPU**: Works on CPU but 10-20x slower
- **Disk**: ~4GB for models
- **RAM**: 8GB+ recommended

### Installation

1. **Clone SadTalker repository**:
```bash
# Clone to models directory
cd models
git clone https://github.com/OpenTalker/SadTalker.git sadtalker
cd sadtalker
```

2. **Install SadTalker dependencies**:
```bash
# Activate your virtual environment first
# Windows:
excusemyfrench\Scripts\activate
# Linux/Mac:
source excusemyfrench/bin/activate

# Install SadTalker requirements
pip install -r requirements.txt
```

3. **Download model checkpoints**:
```bash
# Run SadTalker's download script
python scripts/download_models.py
```

This will download:
- Face detection models (~100MB)
- SadTalker checkpoints (~3.5GB)
- Audio encoder models (~200MB)

4. **Update .env configuration**:
```bash
# Add to config/.env
SADTALKER_CHECKPOINT_PATH=models/sadtalker
LIPSYNC_MODEL=sadtalker
```

### Verify Installation

```bash
# Test SadTalker with example
cd models/sadtalker
python inference.py \
  --driven_audio examples/driven_audio/bus_chinese.wav \
  --source_image examples/source_image/full_body_1.png \
  --result_dir results
```

If successful, you'll see an animated video in `results/`.

---

## Wav2Lip Setup

### What is Wav2Lip?

Wav2Lip is a lightweight lip-sync model that focuses on accurate lip movements. It's faster than SadTalker but doesn't include head pose/expression.

### System Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM (recommended)
- **CPU**: Works on CPU
- **Disk**: ~300MB for models
- **RAM**: 4GB+ recommended

### Installation

1. **Clone Wav2Lip repository**:
```bash
cd models
git clone https://github.com/Rudrabha/Wav2Lip.git wav2lip
cd wav2lip
```

2. **Install Wav2Lip dependencies**:
```bash
# Activate virtual environment
excusemyfrench\Scripts\activate  # Windows
# or
source excusemyfrench/bin/activate  # Linux/Mac

# Install requirements
pip install -r requirements.txt
```

3. **Download model checkpoint**:
```bash
# Download Wav2Lip checkpoint
# Option 1: Direct download
wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?download=1" -O checkpoints/wav2lip.pth

# Option 2: Alternative (Wav2Lip + GAN for better quality)
wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?download=1" -O checkpoints/wav2lip_gan.pth
```

4. **Download face detection model**:
```bash
# Download s3fd face detector
cd face_detection/detection/sfd
wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O s3fd.pth
cd ../../..
```

5. **Update .env configuration**:
```bash
# Add to config/.env
WAV2LIP_CHECKPOINT_PATH=models/wav2lip
LIPSYNC_MODEL=wav2lip
```

### Verify Installation

```bash
# Test Wav2Lip
cd models/wav2lip
python inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face sample_data/main.mp4 \
  --audio sample_data/input_audio.wav \
  --outfile results/result.mp4
```

---

## Wan 2.2 Setup

### What is Wan 2.2?

Wan 2.2 is Alibaba's open-source video generation model supporting:
- **Text-to-Video (T2V)**: Generate videos from text descriptions
- **Image-to-Video (I2V)**: Animate static images
- **720P/480P output** at 24fps
- **5-second videos**

### System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (24GB recommended for 720P)
  - 4090: Can run 5B model at 720P
  - A100/H100: Can run 14B model
- **Disk**: 40-80GB depending on model size
- **RAM**: 32GB+ recommended

### Installation

1. **Install Wan 2.2 dependencies**:
```bash
# Activate virtual environment
excusemyfrench\Scripts\activate  # Windows

pip install diffusers>=0.30.0
pip install transformers>=4.40.0
pip install accelerate>=0.30.0
pip install xformers  # For memory efficiency
```

2. **Set up Hugging Face authentication**:
```bash
# Login to Hugging Face (uses HF_TOKEN from .env)
huggingface-cli login
```

3. **Download Wan 2.2 models**:

**Option A: 5B Model** (Recommended for consumer GPUs)
```python
# Download via Python script
from diffusers import WanPipeline

# Image-to-Video model (recommended for our use case)
pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-5B",
    torch_dtype="float16",
    variant="fp16"
)
pipeline.save_pretrained("models/wan2.2/i2v-5b")

# Text-to-Video model (optional)
pipeline_t2v = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-T2V-5B",
    torch_dtype="float16",
    variant="fp16"
)
pipeline_t2v.save_pretrained("models/wan2.2/t2v-5b")
```

**Option B: 14B Model** (For high-end GPUs only)
```python
from diffusers import WanPipeline

# Requires 24GB+ VRAM
pipeline = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.2-I2V-A14B",
    torch_dtype="float16",
    variant="fp16"
)
pipeline.save_pretrained("models/wan2.2/i2v-14b")
```

4. **Update .env configuration**:
```bash
# Add to config/.env
WAN_MODEL_PATH=models/wan2.2
WAN_MODEL_SIZE=5B  # or 14B
```

### Verify Installation

```python
# Test Wan 2.2 Image-to-Video
from diffusers import WanPipeline
import torch

pipeline = WanPipeline.from_pretrained(
    "models/wan2.2/i2v-5b",
    torch_dtype=torch.float16
).to("cuda")

# Generate 5-second video from image
video = pipeline(
    image="path/to/butcher_image.png",
    prompt="French bulldog looking around curiously",
    num_frames=120,  # 5 seconds at 24fps
    height=720,
    width=480
).frames[0]

# Save video
from diffusers.utils import export_to_video
export_to_video(video, "test_wan_output.mp4", fps=24)
```

---

## Testing Animation

### Test animate.py with SadTalker

```bash
# Generate test audio first
python scripts/generate_audio.py test_data/test_script.json

# Animate with SadTalker
python scripts/animate.py \
  --timeline data/audio/test/timeline.json \
  --images data/image_selections.json \
  --method sadtalker \
  --output data/animated/test
```

### Test single image + audio

```bash
python scripts/animate.py \
  --image data/butcher/butcher_neutral.png \
  --audio data/audio/test/001_butcher_sarcastic.mp3 \
  --output data/test_animated.mp4 \
  --method sadtalker
```

### Compare methods

```bash
# Generate with SadTalker
python scripts/animate.py \
  --image test.png \
  --audio test.mp3 \
  --output sadtalker_result.mp4 \
  --method sadtalker

# Generate with Wav2Lip
python scripts/animate.py \
  --image test.png \
  --audio test.mp3 \
  --output wav2lip_result.mp4 \
  --method wav2lip
```

---

## Troubleshooting

### SadTalker Issues

**"CUDA out of memory"**
- Reduce video resolution in .env: `VIDEO_HEIGHT=1280` (instead of 1920)
- Use CPU: `python animate.py --cpu`
- Close other GPU applications

**"Face not detected"**
- Ensure image has clear frontal face
- Check image resolution (512x512 minimum recommended)
- Try different source image

### Wav2Lip Issues

**"No face detected"**
- Use face detection model: already included in setup
- Ensure clear frontal face in image
- Check lighting and image quality

**"Lips don't match"**
- Wav2Lip works best with clear speech audio
- Avoid background noise in audio
- Try Wav2Lip GAN model for better quality

### Wan 2.2 Issues

**"Insufficient VRAM"**
- Use 5B model instead of 14B
- Reduce resolution: 480P instead of 720P
- Enable gradient checkpointing
- Use CPU offloading (slower)

**"Download failed"**
- Check HF_TOKEN is valid
- Ensure sufficient disk space (40-80GB)
- Try manual download from Hugging Face website

---

## Performance Comparison

| Method | Speed (GPU) | Quality | VRAM | Best For |
|--------|------------|---------|------|----------|
| SadTalker | ~30s/clip | ⭐⭐⭐⭐⭐ | 6GB | Realistic talking heads |
| Wav2Lip | ~10s/clip | ⭐⭐⭐ | 4GB | Fast lip-sync only |
| Wan 2.2 5B | ~60s/clip | ⭐⭐⭐⭐ | 16GB | Creative video generation |
| Wan 2.2 14B | ~120s/clip | ⭐⭐⭐⭐⭐ | 24GB | High-quality video generation |

---

## Next Steps

1. Choose your preferred animation method (SadTalker recommended)
2. Follow installation steps above
3. Test with sample data
4. Integrate into full pipeline
5. Optimize settings for your hardware

For integration into the full Excuse My French pipeline, see `docs/WORKFLOW.md`.
