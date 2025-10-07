# ComfyUI Setup Guide for ExcuseMyFrench

This guide will walk you through setting up ComfyUI with Wan 2.2 models for generating character images in the ExcuseMyFrench video pipeline.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Setup (Automated)](#quick-setup-automated)
- [Manual Setup](#manual-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or higher
- **RAM**: 16GB minimum (32GB recommended)
- **Storage**: 25GB free disk space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (highly recommended)
  - CUDA 11.8 or 12.1 compatible
  - GTX 1080 Ti or better recommended
- **CPU**: Multi-core processor (if no GPU available, expect slower generation)

### Software Dependencies

1. **Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure Python is added to PATH during installation

2. **Git**
   - Download from [git-scm.com](https://git-scm.com/downloads)
   - Required for cloning repositories

3. **Visual C++ Build Tools** (Windows only)
   - Download from [Microsoft](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Required for compiling some Python packages

4. **CUDA Toolkit** (Optional but recommended)
   - Download from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
   - Required for GPU acceleration
   - Version 11.8 or 12.1 recommended

### Checking Prerequisites

Run these commands to verify your system:

```bash
# Check Python version
python --version
# Should show Python 3.10.x or higher

# Check Git
git --version

# Check CUDA (if installed)
nvidia-smi
```

## Quick Setup (Automated)

The easiest way to set up ComfyUI is using our automated setup script.

### Step 1: Navigate to Scripts Directory

```bash
cd D:\ExcuseMyFrench\Repo\ExcuseMyFrench\scripts
```

### Step 2: Run Setup Script

```bash
# Basic setup with defaults
python setup_comfyui.py

# Custom installation path
python setup_comfyui.py --install-path D:/custom/path/ComfyUI

# Force reinstallation
python setup_comfyui.py --force

# Verbose output for debugging
python setup_comfyui.py --verbose
```

### Step 3: Wait for Installation

The script will:
1. Check prerequisites
2. Clone ComfyUI repository
3. Install Python dependencies
4. Download Wan 2.2 models (this may take a while)
5. Install custom nodes
6. Create configuration files
7. Validate installation

**Note**: Model downloads can take 30-60 minutes depending on your internet speed. Each Wan 2.2 model is approximately 6.5GB.

### Step 4: Start ComfyUI Server

After installation completes:

**Windows:**
```bash
D:\ComfyUI\start_comfyui.bat
```

**Linux/Mac:**
```bash
./ComfyUI/start_comfyui.sh
```

The server will start on `http://127.0.0.1:8188`. Open this URL in your browser to access the ComfyUI interface.

## Manual Setup

If you prefer to set up ComfyUI manually or if the automated script fails:

### Step 1: Clone ComfyUI

```bash
cd D:/
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

### Step 2: Install Dependencies

**Windows with CUDA:**
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python -m pip install torch torchvision torchaudio
python -m pip install -r requirements.txt
```

### Step 3: Download Wan 2.2 Models

Models should be placed in `ComfyUI/models/checkpoints/`:

```bash
cd models/checkpoints

# Using wget (Linux/Mac)
wget https://huggingface.co/Wan-AI/Wan-2.2-base/resolve/main/wan_2.2_base.safetensors

# Or using curl
curl -L -o wan_2.2_base.safetensors https://huggingface.co/Wan-AI/Wan-2.2-base/resolve/main/wan_2.2_base.safetensors
```

**Note**: Update the URLs above with actual Wan 2.2 model download links when available.

### Step 4: Install Custom Nodes (Optional)

```bash
cd custom_nodes

# ComfyUI Manager (highly recommended)
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# ControlNet Aux (optional)
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
```

### Step 5: Start ComfyUI

```bash
cd D:/ComfyUI
python main.py --listen 127.0.0.1 --port 8188
```

## Configuration

### Environment Variables

Update your `config/.env` file with ComfyUI settings:

```bash
# ComfyUI Installation
COMFYUI_PATH=D:/ComfyUI
COMFYUI_SERVER_URL=http://127.0.0.1:8188
COMFYUI_OUTPUT_DIR=data/comfyui_output

# Wan 2.2 Models
WAN_MODEL_PATH=models/wan2.2
```

### ComfyUI Settings

Create or edit `ComfyUI/extra_model_paths.yaml`:

```yaml
excusemyfrench:
    base_path: D:/ExcuseMyFrench/Repo/ExcuseMyFrench/models
    checkpoints: wan2.2/
    loras: lora/

output:
    base_path: D:/ExcuseMyFrench/Repo/ExcuseMyFrench/data/comfyui_output
```

### Workflow Files

Pre-configured workflows are located in `comfyui/workflows/`:

- `character_generation.json` - General character generation
- `nutsy_generation.json` - Optimized for Nutsy character

You can load these workflows in ComfyUI's web interface or use them programmatically via the API.

## Usage

### Using the ComfyUI Interface

1. Start the ComfyUI server (see Step 4 above)
2. Open `http://127.0.0.1:8188` in your browser
3. Load a workflow:
   - Click "Load" button
   - Navigate to `comfyui/workflows/character_generation.json`
4. Modify prompts and parameters as needed
5. Click "Queue Prompt" to generate

### Using the Python API

#### Check Connection

```python
from scripts.comfyui_integration import ComfyUIClient

client = ComfyUIClient()
if client.check_connection():
    print("Connected to ComfyUI!")
```

#### Generate Single Image

```python
client = ComfyUIClient()

params = {
    'prompt': 'French bulldog, happy expression, portrait',
    'negative_prompt': 'blurry, low quality',
    'seed': 42,
    'steps': 30,
    'cfg': 7.5
}

images = client.generate_image(
    workflow_path='comfyui/workflows/character_generation.json',
    params=params,
    output_path='output/butcher_happy.png'
)
```

#### Batch Generation

```python
client = ComfyUIClient()

param_list = [
    {'character': 'Butcher', 'emotion': 'happy', 'prompt': '...', 'seed': 42},
    {'character': 'Butcher', 'emotion': 'sad', 'prompt': '...', 'seed': 43},
    {'character': 'Nutsy', 'emotion': 'excited', 'prompt': '...', 'seed': 44},
]

results = client.batch_generate(
    workflow_path='comfyui/workflows/character_generation.json',
    param_list=param_list,
    output_dir='data/generated'
)
```

### Command Line Usage

```bash
# Check connection
python scripts/comfyui_integration.py --check

# Generate image
python scripts/comfyui_integration.py \
    --workflow comfyui/workflows/character_generation.json \
    --prompt "French bulldog, happy expression" \
    --output output.png

# Get system stats
python scripts/comfyui_integration.py --stats

# Get queue status
python scripts/comfyui_integration.py --queue
```

## Troubleshooting

### ComfyUI Won't Start

**Problem**: Server fails to start or crashes immediately.

**Solutions**:
1. Check Python version: `python --version` (must be 3.10+)
2. Verify all dependencies installed:
   ```bash
   cd D:/ComfyUI
   python -m pip install -r requirements.txt
   ```
3. Check for port conflicts:
   ```bash
   # Windows
   netstat -ano | findstr :8188

   # Linux/Mac
   lsof -i :8188
   ```
4. Try running with more verbose output:
   ```bash
   python main.py --listen 127.0.0.1 --port 8188 --verbose
   ```

### CUDA Out of Memory

**Problem**: "CUDA out of memory" error during generation.

**Solutions**:
1. Reduce image resolution in workflow (e.g., 512x512 instead of 1080x1920)
2. Enable CPU offloading in ComfyUI settings
3. Reduce batch size to 1
4. Close other GPU-intensive applications
5. Use `--lowvram` or `--novram` flags:
   ```bash
   python main.py --listen 127.0.0.1 --port 8188 --lowvram
   ```

### Models Not Loading

**Problem**: Checkpoint not found or fails to load.

**Solutions**:
1. Verify model files exist:
   ```bash
   ls D:/ComfyUI/models/checkpoints/
   ```
2. Check file isn't corrupted (should be ~6.5GB):
   ```bash
   # Windows PowerShell
   Get-Item D:/ComfyUI/models/checkpoints/wan_2.2_base.safetensors | Select-Object Length
   ```
3. Re-download the model if necessary
4. Ensure file permissions allow reading
5. Check `extra_model_paths.yaml` configuration

### Generation is Very Slow

**Problem**: Image generation takes several minutes.

**Solutions**:
1. Verify GPU is being used:
   ```bash
   nvidia-smi
   ```
   You should see Python process using GPU
2. Install CUDA-enabled PyTorch:
   ```bash
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
3. Reduce number of inference steps (20-25 is usually sufficient)
4. Use faster samplers (euler, euler_ancestral)
5. Enable xformers for memory optimization

### Connection Refused

**Problem**: Cannot connect to ComfyUI API.

**Solutions**:
1. Ensure server is running:
   ```bash
   # Check if process is running
   # Windows
   tasklist | findstr python

   # Linux/Mac
   ps aux | grep python
   ```
2. Verify server URL in `.env`:
   ```
   COMFYUI_SERVER_URL=http://127.0.0.1:8188
   ```
3. Try accessing in browser: `http://127.0.0.1:8188`
4. Check firewall settings
5. Restart the server

### Workflow Errors

**Problem**: Workflow fails with "Node not found" or similar errors.

**Solutions**:
1. Ensure all custom nodes are installed
2. Update ComfyUI to latest version:
   ```bash
   cd D:/ComfyUI
   git pull
   ```
3. Reload custom nodes in ComfyUI Manager
4. Check workflow file is valid JSON
5. Verify all referenced models exist

## Advanced Topics

### Custom Workflows

You can create custom workflows in the ComfyUI interface:

1. Design your workflow in the web UI
2. Click "Save" to export as JSON
3. Save to `comfyui/workflows/` directory
4. Use with the API or CLI

### Emotion-Specific Prompts

Customize prompts for different emotions:

```python
emotion_prompts = {
    'happy': 'smiling, joyful, cheerful expression',
    'sad': 'sad, downcast, melancholy expression',
    'excited': 'excited, energetic, enthusiastic expression',
    'sarcastic': 'smirking, knowing look, raised eyebrow',
    'angry': 'angry, fierce, furrowed brow',
    'confused': 'confused, puzzled, questioning look',
}

for emotion, emotion_desc in emotion_prompts.items():
    params = {
        'prompt': f'French bulldog, {emotion_desc}, portrait, high quality',
        'seed': random.randint(1, 1000000),
    }
    # Generate...
```

### Performance Optimization

#### Use Multiple GPUs

Edit `ComfyUI/extra_model_paths.yaml`:

```yaml
gpu:
    device: 0  # Use first GPU
    # device: 1  # Use second GPU
```

#### Enable xformers

```bash
python -m pip install xformers
```

Then restart ComfyUI - it will automatically detect and use xformers.

#### Model Caching

Keep ComfyUI server running between generations to avoid reloading models:

```python
# Create a long-lived client
client = ComfyUIClient()

# Generate multiple times without restarting server
for params in param_list:
    images = client.generate_image(workflow_path, params)
```

### Integration with Video Pipeline

ComfyUI integrates seamlessly with the ExcuseMyFrench pipeline:

```python
# In your video generation script
from scripts.comfyui_integration import ComfyUIClient
from scripts.select_images import ImageSelector

# Generate missing images
client = ComfyUIClient()
selector = ImageSelector()

missing = selector.check_missing_images(script)

for item in missing:
    params = {
        'character': item['character'],
        'emotion': item['emotion'],
        'prompt': f"{item['character']} with {item['emotion']} expression",
    }

    images = client.generate_image(
        workflow_path='comfyui/workflows/character_generation.json',
        params=params,
        output_path=f"data/generated/{item['character']}_{item['emotion']}.png"
    )

    # Add to library
    selector.add_image(
        character=item['character'],
        emotion=item['emotion'],
        file_path=output_path,
        source='comfyui',
        quality_score=0.85
    )
```

### Remote ComfyUI Server

To run ComfyUI on a remote server:

1. Start ComfyUI with public access:
   ```bash
   python main.py --listen 0.0.0.0 --port 8188
   ```

2. Update `.env`:
   ```
   COMFYUI_SERVER_URL=http://your-server-ip:8188
   ```

3. Ensure firewall allows port 8188

**Security Note**: Never expose ComfyUI to the public internet without authentication!

### Model Management

Keep models organized:

```
ComfyUI/
├── models/
│   ├── checkpoints/
│   │   ├── wan_2.2_base.safetensors
│   │   └── wan_2.2_anime.safetensors
│   ├── loras/
│   │   └── butcher_lora.safetensors
│   ├── vae/
│   └── embeddings/
```

### Monitoring and Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)

client = ComfyUIClient()
# Will now show detailed API calls and responses
```

Monitor system resources:

```python
stats = client.get_system_stats()
print(f"VRAM Usage: {stats['vram_used']} / {stats['vram_total']}")
print(f"Queue Size: {len(stats['queue_running'])}")
```

## Additional Resources

- [ComfyUI Official Repository](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI/wiki)
- [Wan 2.2 Model Documentation](https://huggingface.co/Wan-AI) (update with actual link)
- [ExcuseMyFrench Project README](../README.md)
- [Image Generation Workflow](IMAGE_GENERATION.md)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [ComfyUI Issues](https://github.com/comfyanonymous/ComfyUI/issues)
2. Review ExcuseMyFrench project documentation
3. Enable verbose logging for debugging
4. Check system resources (RAM, VRAM, disk space)

## Next Steps

After setting up ComfyUI:

1. Test the installation with sample generations
2. Review and customize the workflow files
3. Integrate with the main video pipeline
4. Generate character images for your library
5. Start creating videos!

See [WORKFLOW.md](WORKFLOW.md) for the complete video generation pipeline.
