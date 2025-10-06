# DreamBooth Training Guide

This guide covers training a custom DreamBooth model for Butcher character to ensure consistent appearance across all generated images.

## Table of Contents

- [Overview](#overview)
- [Preparing Training Data](#preparing-training-data)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Using the Model](#using-the-model)

---

## Overview

### What is DreamBooth?

DreamBooth is a technique to fine-tune Stable Diffusion models on a specific subject (in our case, Butcher the French bulldog) using only 5-15 training images. This ensures consistent character appearance across different emotions and poses.

### Why Use DreamBooth?

**Without DreamBooth:**
- Each generated image looks like a different dog
- Inconsistent colors, markings, features
- Hard to maintain character identity

**With DreamBooth:**
- Every image clearly shows the same character (Butcher)
- Consistent appearance across emotions
- Professional, recognizable brand identity

### System Requirements

**Minimum (Training will be slow):**
- GPU: NVIDIA with 12GB VRAM (GTX 3060, RTX 3080)
- RAM: 16GB
- Disk: 20GB free
- Time: 2-4 hours

**Recommended:**
- GPU: NVIDIA with 24GB VRAM (RTX 3090, RTX 4090, A100)
- RAM: 32GB
- Disk: 40GB free
- Time: 40-60 minutes

**Note**: Training on CPU is possible but NOT recommended (20-50x slower).

---

## Preparing Training Data

### Collecting Butcher Images

You need **5-15 high-quality images** of Butcher for training.

**Image Requirements:**

✅ **Good Training Images:**
- Clear, well-lit photos of Butcher
- Different angles (front, side, slight angles)
- Different lighting conditions
- Different backgrounds
- Natural poses
- High resolution (512x512 minimum, 1024x1024 better)
- Butcher is the main subject (full body or portrait)

❌ **Bad Training Images:**
- Blurry or low-resolution images
- Multiple dogs in one image
- Heavily filtered or edited photos
- Extreme angles or crops
- Butcher is very small in frame
- Duplicate or nearly identical images

### Organizing Training Data

1. **Create training directory**:
```bash
mkdir -p data/butcher_training
```

2. **Copy Butcher images**:
```bash
# Copy your best 5-15 images of Butcher
cp /path/to/butcher/photos/* data/butcher_training/
```

3. **Verify images**:
```bash
ls -lh data/butcher_training/
# Should show 5-15 .jpg or .png files
```

**Directory Structure:**
```
data/
└── butcher_training/
    ├── butcher_001.jpg
    ├── butcher_002.jpg
    ├── butcher_003.jpg
    ├── butcher_004.jpg
    ├── butcher_005.jpg
    └── ... (up to 15 images)
```

### Image Preprocessing (Optional)

For best results, preprocess images:

**Resize to 512x512**:
```python
from PIL import Image
from pathlib import Path

source_dir = Path("data/butcher_raw")
target_dir = Path("data/butcher_training")
target_dir.mkdir(exist_ok=True)

for img_path in source_dir.glob("*.jpg"):
    img = Image.open(img_path)
    # Resize to 512x512
    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
    # Save
    img_resized.save(target_dir / img_path.name)
```

**Center crop faces** (if images are full-body):
```python
# Use face detection to crop around face
# See scripts/preprocess_training_images.py (optional script)
```

---

## Installation

### Install DreamBooth Training Dependencies

1. **Activate virtual environment**:
```bash
# Windows:
excusemyfrench\Scripts\activate

# Linux/Mac:
source excusemyfrench/bin/activate
```

2. **Install required packages**:
```bash
pip install accelerate==0.30.1
pip install diffusers[torch]==0.30.0
pip install transformers==4.40.0
pip install xformers  # Optional, for memory efficiency
pip install bitsandbytes  # Optional, for 8-bit Adam optimizer
```

3. **Configure accelerate**:
```bash
accelerate config
```

Answer prompts:
- Compute environment: **This machine**
- Machine type: **No distributed training**
- Use DeepSpeed: **No**
- Use FullyShardedDataParallel: **No**
- GPU: **Select your GPU count** (usually 1)
- Mixed precision: **fp16** (or bf16 if supported)

### Download Hugging Face Training Script

```bash
# Create training scripts directory
mkdir -p training_scripts
cd training_scripts

# Download DreamBooth training script
wget https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth.py

cd ..
```

**Alternative**: Clone full diffusers repository:
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers/examples/dreambooth
pip install -r requirements.txt
cd ../../..
```

---

## Training the Model

### Basic Training (No Prior Preservation)

Simplest method - train on Butcher images only:

```bash
python scripts/train_dreambooth.py \
  --instance-data data/butcher_training \
  --output models/dreambooth_butcher \
  --instance-prompt "a photo of sks dog" \
  --steps 800 \
  --learning-rate 5e-6
```

**Parameters:**
- `--instance-data`: Directory with Butcher training images
- `--output`: Where to save trained model
- `--instance-prompt`: Special prompt identifier ("sks" is common convention)
- `--steps`: Number of training steps (800 typical, 400-1200 range)
- `--learning-rate`: Learning rate (5e-6 typical)

### Advanced Training (With Prior Preservation)

Better results - prevents "overfitting" and maintains general dog knowledge:

```bash
python scripts/train_dreambooth.py \
  --instance-data data/butcher_training \
  --class-data data/class_dogs \
  --output models/dreambooth_butcher \
  --instance-prompt "a photo of sks dog" \
  --class-prompt "a photo of a dog" \
  --prior-preservation \
  --steps 1200 \
  --learning-rate 2e-6
```

**Additional Parameters:**
- `--class-data`: Directory for regularization images (auto-generated if doesn't exist)
- `--class-prompt`: Prompt for regularization images
- `--prior-preservation`: Enable prior preservation loss

### Memory-Optimized Training

For GPUs with limited VRAM (12GB):

```bash
python scripts/train_dreambooth.py \
  --instance-data data/butcher_training \
  --output models/dreambooth_butcher \
  --instance-prompt "a photo of sks dog" \
  --steps 800 \
  --learning-rate 5e-6 \
  --resolution 512 \
  --batch-size 1 \
  --use-8bit-adam \
  --mixed-precision fp16
```

**Memory-Saving Options:**
- `--resolution 512`: Train at 512x512 instead of 768
- `--batch-size 1`: Process one image at a time
- `--use-8bit-adam`: Use memory-efficient optimizer
- `--mixed-precision fp16`: Use half-precision training

### High-Quality Training

For GPUs with ample VRAM (24GB+):

```bash
python scripts/train_dreambooth.py \
  --instance-data data/butcher_training \
  --class-data data/class_dogs \
  --output models/dreambooth_butcher \
  --instance-prompt "a photo of sks dog" \
  --class-prompt "a photo of a dog" \
  --prior-preservation \
  --steps 1200 \
  --learning-rate 2e-6 \
  --resolution 768 \
  --batch-size 2 \
  --mixed-precision bf16
```

### Training Progress

During training, you'll see:

```
Step 100/1200 | Loss: 0.0523
Step 200/1200 | Loss: 0.0418
Step 300/1200 | Loss: 0.0372
...
✓ Training completed!
Model saved to: models/dreambooth_butcher
```

**What to Look For:**
- Loss should decrease over time
- Typical final loss: 0.01-0.05
- Too low loss (<0.005): May be overfitting
- High loss (>0.10): Needs more training

**Typical Training Time:**
- **RTX 4090 (24GB)**: 40-60 minutes @ 768px
- **RTX 3090 (24GB)**: 60-90 minutes @ 768px
- **RTX 3080 (12GB)**: 90-150 minutes @ 512px
- **GTX 3060 (12GB)**: 120-200 minutes @ 512px

---

## Testing the Model

### Generate Test Images

After training completes, test the model:

```bash
python scripts/train_dreambooth.py \
  --test models/dreambooth_butcher \
  --test-output data/test_outputs
```

This generates 4 test images with different emotions.

### Manual Testing

Generate custom test images:

```python
from diffusers import StableDiffusionPipeline
import torch

# Load trained model
pipeline = StableDiffusionPipeline.from_pretrained(
    "models/dreambooth_butcher",
    torch_dtype=torch.float16
).to("cuda")

# Generate image
image = pipeline(
    "a photo of sks dog, sarcastic expression, professional photography",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("test_butcher_sarcastic.png")
```

### Evaluating Results

**Good Results:**
✅ Images clearly show Butcher's distinctive features
✅ Consistent appearance across different emotions
✅ Natural-looking results
✅ Recognizable as the same character

**Poor Results (Need Retraining):**
❌ Generic dog, doesn't look like Butcher
❌ Strange artifacts or distortions
❌ Overfitted (looks identical to training images)
❌ Lost ability to generate different poses/emotions

### Troubleshooting Poor Results

**If images don't look like Butcher:**
- Add more training images (aim for 10-15)
- Increase training steps (try 1200-1600)
- Adjust learning rate (try 2e-6 or 1e-5)
- Use prior preservation

**If images are overfitted:**
- Reduce training steps (try 400-600)
- Lower learning rate (try 2e-6 or 1e-6)
- Add more diversity in training images

**If images have artifacts:**
- Check training image quality
- Reduce learning rate
- Use gradient checkpointing
- Try different random seed

---

## Using the Model

### Update Configuration

1. **Update .env file**:
```bash
# Edit config/.env
DREAMBOOTH_MODEL_PATH=models/dreambooth_butcher
```

2. **Verify path**:
```bash
ls -lh models/dreambooth_butcher/
# Should show model files (model_index.json, unet/, vae/, etc.)
```

### Generate Images with Trained Model

The model is now automatically used by `generate_images.py`:

```bash
# Generate Butcher image with specific emotion
python scripts/generate_images.py \
  --character Butcher \
  --emotion sarcastic

# Generate all emotions
for emotion in happy sad excited sarcastic angry neutral; do
  python scripts/generate_images.py \
    --character Butcher \
    --emotion $emotion
done
```

### Prompt Format

When using the trained model, always include the special identifier:

```python
# Correct prompts:
"a photo of sks dog, happy expression, ..."
"a photo of sks dog, French bulldog, sarcastic, ..."
"sks dog looking excited, portrait, ..."

# Incorrect (won't use trained model):
"a photo of a French bulldog, happy"  # Missing "sks dog"
"a bulldog with sarcastic expression"  # Missing "sks dog"
```

---

## Advanced Topics

### LoRA Training (Alternative)

LoRA is a lighter alternative to full DreamBooth:

**Advantages:**
- Smaller file size (~5MB vs ~4GB)
- Faster training (20-40 minutes)
- Can combine multiple LoRAs

**Disadvantages:**
- Slightly less consistent than full DreamBooth
- Requires careful weight tuning

**Training LoRA:**
```bash
# Use Hugging Face LoRA training script
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --instance_data_dir="data/butcher_training" \
  --output_dir="models/lora_butcher" \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=1e-4 \
  --max_train_steps=800
```

### Combining with ControlNet

Use ControlNet for precise pose control:

1. Train DreamBooth for appearance
2. Use ControlNet for pose guidance
3. Combine both for maximum control

### Multi-Concept Training

Train both Butcher and Nutsy in one model:

```bash
# Train with two instance directories
accelerate launch train_dreambooth.py \
  --instance_data_dir_1="data/butcher_training" \
  --instance_prompt_1="a photo of sks dog" \
  --instance_data_dir_2="data/nutsy_training" \
  --instance_prompt_2="a photo of xyz squirrel" \
  ...
```

---

## Environment Variables

Configure training in `config/.env`:

```bash
# DreamBooth Training Settings
DREAMBOOTH_TRAIN_STEPS=800          # Training steps
DREAMBOOTH_LEARNING_RATE=5e-6       # Learning rate
DREAMBOOTH_BATCH_SIZE=1             # Batch size
DREAMBOOTH_GRADIENT_ACCUM=1         # Gradient accumulation
DREAMBOOTH_RESOLUTION=512           # Training resolution

# Model Paths
DREAMBOOTH_MODEL_PATH=models/dreambooth_butcher
STABLE_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
```

---

## Best Practices

### Training Data

- **Quantity**: 8-12 images is the sweet spot
- **Quality**: High resolution, well-lit, clear
- **Diversity**: Different angles, poses, lighting
- **Consistency**: All images should clearly show the same subject

### Training Parameters

- **Steps**: Start with 800, adjust based on results
- **Learning Rate**: 5e-6 is safe, 2e-6 for fine detail
- **Resolution**: 512px for speed, 768px for quality
- **Prior Preservation**: Use it for better results

### Validation

- Test model on unseen prompts
- Generate various emotions and poses
- Compare to original training images
- Ensure character consistency

---

## Troubleshooting

### "CUDA out of memory"

- Reduce resolution to 512
- Use batch size 1
- Enable gradient checkpointing
- Use 8-bit Adam optimizer
- Try gradient accumulation

### "Training taking too long"

- Reduce training steps
- Use smaller resolution (512)
- Disable prior preservation
- Use fewer class images

### "Model not loading"

- Check model path is correct
- Verify all model files exist
- Check diffusers version compatibility
- Try loading with different dtype (float32 vs float16)

---

## Next Steps

1. Collect 8-12 high-quality Butcher images
2. Organize into `data/butcher_training/`
3. Install DreamBooth training dependencies
4. Run training script (start with 800 steps)
5. Test generated images
6. Adjust parameters if needed
7. Retrain with better settings
8. Update .env with model path
9. Use in pipeline

For more information:
- DreamBooth Paper: https://dreambooth.github.io/
- Hugging Face Guide: https://huggingface.co/docs/diffusers/training/dreambooth
