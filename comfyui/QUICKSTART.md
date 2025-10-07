# ComfyUI Quick Start Guide

Get ComfyUI up and running in 5 minutes!

## 1. Prerequisites Check

Make sure you have:
- Python 3.10+ installed
- Git installed
- 25GB free disk space
- (Optional) NVIDIA GPU with 8GB+ VRAM

```bash
python --version  # Should show 3.10 or higher
git --version     # Should show git version
```

## 2. Install ComfyUI

Run our automated setup script:

```bash
cd scripts
python setup_comfyui.py
```

This will:
- Clone ComfyUI
- Install all dependencies
- Download Wan 2.2 models
- Configure everything

**Note**: This takes 30-60 minutes. Go grab coffee!

## 3. Start ComfyUI Server

### Windows
```bash
D:\ComfyUI\start_comfyui.bat
```

### Linux/Mac
```bash
./ComfyUI/start_comfyui.sh
```

Wait for: `Starting server`

Open: http://127.0.0.1:8188

## 4. Test Your Setup

```bash
# Quick test (no image generation)
python scripts/test_comfyui.py

# Full test (includes image generation)
python scripts/test_comfyui.py --generate
```

## 5. Generate Your First Image

### Option A: Python API

```python
from scripts.comfyui_integration import ComfyUIClient

client = ComfyUIClient()

images = client.generate_image(
    workflow_path='comfyui/workflows/character_generation.json',
    params={
        'prompt': 'French bulldog, happy, portrait',
        'seed': 42
    },
    output_path='my_first_image.png'
)
```

### Option B: Command Line

```bash
python scripts/comfyui_integration.py \
    --workflow comfyui/workflows/character_generation.json \
    --prompt "French bulldog, happy expression" \
    --output my_first_image.png
```

### Option C: Web Interface

1. Open http://127.0.0.1:8188
2. Click "Load"
3. Select `comfyui/workflows/character_generation.json`
4. Edit the prompt
5. Click "Queue Prompt"

## Common Issues

### "Connection refused"
→ Make sure ComfyUI server is running

### "CUDA out of memory"
→ Start server with: `python main.py --lowvram`

### "Model not found"
→ Check models exist in `ComfyUI/models/checkpoints/`

### "Generation is slow"
→ Normal on CPU. GPU recommended for production use.

## Next Steps

- Read [full setup guide](../docs/comfyui_setup.md) for advanced options
- Explore workflow files in `comfyui/workflows/`
- Integrate with video pipeline
- Generate character image library

## Need Help?

- See [troubleshooting guide](../docs/comfyui_setup.md#troubleshooting)
- Check ComfyUI server logs
- Enable verbose mode: `--verbose`

## Summary

```bash
# 1. Setup (once)
python scripts/setup_comfyui.py

# 2. Start server (each session)
D:\ComfyUI\start_comfyui.bat

# 3. Test
python scripts/test_comfyui.py --generate

# 4. Generate!
python scripts/comfyui_integration.py \
    --workflow comfyui/workflows/character_generation.json \
    --prompt "your prompt here" \
    --output output.png
```

That's it! You're ready to generate character images with ComfyUI.
