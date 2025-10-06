# ComfyUI Setup and Integration

This guide covers setting up ComfyUI for the Excuse My French pipeline and creating custom workflows for character generation and animation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Integration with Pipeline](#integration-with-pipeline)
- [Custom Workflows](#custom-workflows)
- [API Usage](#api-usage)

---

## Overview

### What is ComfyUI?

ComfyUI is a powerful node-based interface for Stable Diffusion and other AI models. It allows you to create complex workflows visually and run them via API.

### Why Use ComfyUI?

For Excuse My French, ComfyUI provides:
- **Visual workflow creation** for character image generation
- **Advanced control** over Stable Diffusion parameters
- **Batch processing** capabilities
- **Custom nodes** for specialized tasks (LoRA, ControlNet, etc.)
- **API access** for automation

---

## Installation

### Option 1: Standalone Installation (Recommended)

**System Requirements:**
- Windows 10/11, macOS, or Linux
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 20GB+ free disk space
- Python 3.10+

**Installation Steps:**

1. **Download ComfyUI**:
```bash
# Clone ComfyUI repository
cd D:\ExcuseMyFrench\Repo
git clone https://github.com/comfyanonymous/ComfyUI.git

# Or download portable version (Windows):
# https://github.com/comfyanonymous/ComfyUI/releases
```

2. **Install dependencies**:
```bash
cd ComfyUI
# Windows with NVIDIA GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# macOS (Apple Silicon):
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

3. **Download base models**:
```bash
# Download Stable Diffusion 1.5 to ComfyUI/models/checkpoints/
cd models/checkpoints
wget https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors
```

4. **Test installation**:
```bash
cd ../..
python main.py

# ComfyUI should start and open browser at http://127.0.0.1:8188
```

### Option 2: Docker Installation

**Prerequisites:**
- Docker Desktop installed
- NVIDIA GPU with nvidia-docker support (optional but recommended)

**Installation Steps:**

1. **Pull ComfyUI Docker image**:
```bash
docker pull yanwk/comfyui-boot:latest
```

2. **Create docker-compose.yml**:
```yaml
# Create: D:\ExcuseMyFrench\Repo\ExcuseMyFrench\docker-compose.yml
version: '3.8'

services:
  comfyui:
    image: yanwk/comfyui-boot:latest
    ports:
      - "8188:8188"
    volumes:
      - ./ComfyUI/models:/app/models
      - ./ComfyUI/input:/app/input
      - ./ComfyUI/output:/app/output
      - ./workflows/comfyui:/app/workflows
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. **Start ComfyUI**:
```bash
cd D:\ExcuseMyFrench\Repo\ExcuseMyFrench
docker-compose up -d
```

4. **Access ComfyUI**:
Open browser to http://localhost:8188

---

## Integration with Pipeline

### Directory Structure

```
ExcuseMyFrench/
├── ComfyUI/                    # ComfyUI installation (if standalone)
│   ├── models/
│   │   ├── checkpoints/       # SD models
│   │   ├── loras/             # LoRA models
│   │   └── controlnet/        # ControlNet models
│   ├── input/                 # Input images
│   └── output/                # Generated images
├── workflows/
│   └── comfyui/               # Custom workflows for ExcuseMyFrench
│       ├── butcher_generation.json
│       ├── nutsy_generation.json
│       └── emotion_variation.json
└── scripts/
    └── comfyui_generate.py    # Python script to call ComfyUI API
```

### Link Models to ComfyUI

To use your existing models with ComfyUI:

**Windows (Symlinks):**
```bash
# Link Dreambooth Butcher model
mklink /D "D:\ExcuseMyFrench\Repo\ComfyUI\models\checkpoints\butcher_dreambooth" "D:\ExcuseMyFrench\Repo\ExcuseMyFrench\models\dreambooth_butcher"

# Link LoRA models
mklink /D "D:\ExcuseMyFrench\Repo\ComfyUI\models\loras\excusemyfrench" "D:\ExcuseMyFrench\Repo\ExcuseMyFrench\models\lora"
```

**Linux/macOS:**
```bash
# Link Dreambooth Butcher model
ln -s ~/ExcuseMyFrench/models/dreambooth_butcher ~/ComfyUI/models/checkpoints/butcher_dreambooth

# Link LoRA models
ln -s ~/ExcuseMyFrench/models/lora ~/ComfyUI/models/loras/excusemyfrench
```

---

## Custom Workflows

### Butcher Character Generation Workflow

This workflow generates Butcher images with specific emotions using the DreamBooth model.

**Create: `workflows/comfyui/butcher_generation.json`**

1. **Open ComfyUI** (http://127.0.0.1:8188)

2. **Create workflow**:
   - Load Checkpoint: `butcher_dreambooth` or SD 1.5
   - CLIP Text Encode (Positive):
     ```
     a photo of sks dog, French bulldog, {emotion} expression,
     portrait, high quality, detailed, professional photography,
     clear face, front view, looking at camera
     ```
   - CLIP Text Encode (Negative):
     ```
     blurry, low quality, distorted, deformed, ugly, bad anatomy,
     extra limbs, missing limbs, watermark, text, multiple dogs
     ```
   - Empty Latent Image: 1080x1920 (vertical)
   - KSampler:
     - Steps: 50
     - CFG: 7.5
     - Sampler: euler_a
     - Scheduler: normal
   - VAE Decode
   - Save Image: `butcher_{emotion}_{seed}.png`

3. **Save workflow**: Menu → Save → `butcher_generation.json`

4. **Export API format**: Menu → Save (API Format) → `butcher_generation_api.json`

### Nutsy Character Generation Workflow

Similar to Butcher but uses base SD model with different prompt:

**Prompt:**
```
a cute squirrel character, {emotion} expression,
cartoon style, anthropomorphic, expressive face,
portrait, high quality, detailed, clear face, front view
```

### Emotion Variation Workflow

Batch generate all emotions for a character:

- Use Prompt Scheduler node to cycle through emotions
- Batch size: 8 (one for each emotion)
- Seeds: Different for variety

---

## API Usage

### Python Integration

**Create: `scripts/comfyui_generate.py`**

```python
#!/usr/bin/env python3
"""
Generate images using ComfyUI API.
"""

import json
import requests
import time
from pathlib import Path

class ComfyUIClient:
    """Client for ComfyUI API."""

    def __init__(self, url: str = "http://127.0.0.1:8188"):
        self.url = url

    def load_workflow(self, workflow_path: str) -> dict:
        """Load workflow JSON."""
        with open(workflow_path, 'r') as f:
            return json.load(f)

    def queue_prompt(self, workflow: dict) -> str:
        """Queue a prompt for generation."""
        response = requests.post(
            f"{self.url}/prompt",
            json={"prompt": workflow}
        )
        return response.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """Get generation history."""
        response = requests.get(f"{self.url}/history/{prompt_id}")
        return response.json()

    def wait_for_completion(self, prompt_id: str, timeout: int = 300):
        """Wait for generation to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)

            if prompt_id in history:
                if history[prompt_id]["status"]["completed"]:
                    return history[prompt_id]

            time.sleep(1)

        raise TimeoutError("Generation timed out")

    def generate_image(
        self,
        workflow_path: str,
        emotion: str,
        character: str = "Butcher",
        seed: int = None
    ) -> str:
        """
        Generate image with ComfyUI.

        Args:
            workflow_path: Path to workflow JSON
            emotion: Emotion to generate
            character: Character name
            seed: Random seed (None for random)

        Returns:
            Path to generated image
        """
        # Load workflow
        workflow = self.load_workflow(workflow_path)

        # Update parameters (node IDs may vary)
        # Find and update text prompts, seeds, etc.
        # This depends on your specific workflow structure

        # Queue prompt
        prompt_id = self.queue_prompt(workflow)

        # Wait for completion
        result = self.wait_for_completion(prompt_id)

        # Get output image path
        outputs = result["outputs"]
        # Parse outputs to get image path

        return image_path


# Example usage
if __name__ == "__main__":
    client = ComfyUIClient()

    image_path = client.generate_image(
        workflow_path="workflows/comfyui/butcher_generation_api.json",
        emotion="sarcastic",
        character="Butcher"
    )

    print(f"Generated: {image_path}")
```

### Integration with generate_images.py

Modify `scripts/generate_images.py` to support ComfyUI as alternative backend:

```python
# In generate_images.py, add:

def generate_image_comfyui(
    character: str,
    emotion: str,
    output_path: str
) -> str:
    """Generate image using ComfyUI."""
    from comfyui_generate import ComfyUIClient

    client = ComfyUIClient(
        url=os.getenv("COMFYUI_API_URL", "http://127.0.0.1:8188")
    )

    workflow_path = f"workflows/comfyui/{character.lower()}_generation_api.json"

    return client.generate_image(
        workflow_path=workflow_path,
        emotion=emotion,
        character=character
    )
```

---

## Environment Configuration

Add to `config/.env`:

```bash
# ComfyUI Configuration
COMFYUI_PATH=D:/ExcuseMyFrench/Repo/ComfyUI
COMFYUI_API_URL=http://127.0.0.1:8188
COMFYUI_TIMEOUT=300

# Image generation backend
IMAGE_GENERATION_BACKEND=comfyui  # or 'diffusers' for direct Python
```

---

## Custom Nodes (Optional)

### Useful Custom Nodes for Excuse My French

1. **ComfyUI Manager** (for installing other nodes):
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

2. **ControlNet Nodes** (for pose control):
```bash
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
```

3. **Prompt Scheduler** (for batch emotions):
```bash
git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git
```

4. **Video Nodes** (for Wan 2.2 integration):
```bash
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
```

---

## Troubleshooting

### ComfyUI Won't Start

- Check Python version: `python --version` (should be 3.10+)
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Check port 8188 isn't already in use

### Out of Memory Errors

- Reduce batch size in workflow
- Use `--lowvram` flag: `python main.py --lowvram`
- Close other GPU applications
- Use smaller checkpoint models

### Slow Generation

- Enable xformers: `pip install xformers`
- Update ComfyUI: `git pull` in ComfyUI directory
- Check GPU utilization during generation

---

## Next Steps

1. Install ComfyUI (standalone or Docker)
2. Download base models
3. Create custom workflows for Butcher and Nutsy
4. Test API integration
5. Update generate_images.py to use ComfyUI backend
6. Create batch generation scripts

For more information, see:
- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- ComfyUI Examples: https://comfyanonymous.github.io/ComfyUI_examples/
