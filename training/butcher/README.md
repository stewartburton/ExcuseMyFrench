# Butcher the French Bulldog - DreamBooth Training Data

This directory contains training data for DreamBooth fine-tuning to create a custom Stable Diffusion model for Butcher, the young French Bulldog character.

## Directory Structure

```
training/butcher/
├── README.md              # This file
├── images/                # Training images (10-20 photos of Butcher)
│   ├── butcher_0001.png
│   ├── butcher_0002.png
│   └── ...
├── class_images/          # Prior preservation images (auto-generated)
│   └── (generated dog images)
└── captions.txt          # Image captions (auto-generated)
```

## Adding Training Images

### Requirements

For best results, your training images should:

1. **Quantity**: 10-20 high-quality images
   - Minimum: 5 images
   - Recommended: 15-20 images
   - More images = better consistency

2. **Quality**:
   - Resolution: At least 512x512 pixels (higher is better)
   - Clear, well-lit photos
   - Sharp focus (not blurry)
   - Good exposure (not too dark or bright)

3. **Variety**:
   - Different angles (front, side, three-quarter view)
   - Different expressions (happy, neutral, curious, etc.)
   - Different poses (sitting, standing, lying down)
   - Consistent subject (same dog - Butcher)
   - Varied backgrounds (helps generalization)

4. **Avoid**:
   - Multiple dogs in the same photo
   - Heavy filters or effects
   - Watermarks or text overlays
   - Very dark or overexposed images
   - Motion blur

### Recommended Photo Types

For Butcher the French Bulldog, gather photos showing:

- **Front face**: Clear view of face, looking at camera
- **Profile**: Side view showing body shape
- **Happy expression**: Mouth open, tongue out, relaxed
- **Neutral expression**: Calm, natural look
- **Different lighting**: Indoor and outdoor
- **Different contexts**: On furniture, in yard, etc.

### How to Add Images

1. **Collect Photos**: Gather 10-20 photos of Butcher following the guidelines above

2. **Prepare Images**:
   ```bash
   # Option A: Use the automated preparation script
   python scripts/prepare_training_data.py \
     --input path/to/raw/photos \
     --output training/butcher/images \
     --character butcher \
     --class-name "french bulldog"

   # Option B: Manual preparation
   # - Resize images to at least 512x512
   # - Convert to PNG or JPEG
   # - Copy to training/butcher/images/
   ```

3. **Validate Dataset**:
   ```bash
   python scripts/prepare_training_data.py \
     --validate \
     --dataset training/butcher/images
   ```

## Training Configuration

The training configuration for Butcher is stored in `training/config/butcher_config.yaml`.

Key parameters:
- **Instance prompt**: `"a photo of sks dog"` (the unique identifier "sks" represents Butcher)
- **Class prompt**: `"a photo of a french bulldog"` (for prior preservation)
- **Training steps**: 800-1200 (adjust based on dataset size)
- **Learning rate**: 5e-6 (conservative for stability)

## Training Process

### 1. Prepare Data

```bash
# If not already done, prepare your training images
python scripts/prepare_training_data.py \
  --input photos/butcher \
  --output training/butcher/images \
  --character butcher \
  --class-name "french bulldog"
```

### 2. Run Training

```bash
# Using config file (recommended)
python scripts/train_dreambooth.py \
  --config training/config/butcher_config.yaml

# Or with command-line options
python scripts/train_dreambooth.py \
  --character butcher \
  --steps 1000 \
  --lr 5e-6
```

### 3. Monitor Training

Training progress will be saved to:
- **Checkpoints**: `models/dreambooth_butcher/checkpoint-XXX/`
- **Logs**: `logs/dreambooth/`
- **Validation images**: `models/dreambooth_butcher/validation-XXX/`

### 4. Test the Model

```bash
# Generate test images
python scripts/generate_character_image.py \
  --character butcher \
  --emotion happy \
  --count 5
```

## Expected Training Time

With a modern GPU:
- **800 steps**: ~15-30 minutes
- **1200 steps**: ~25-45 minutes

Without GPU (CPU only):
- **800 steps**: ~3-6 hours
- Not recommended for regular use

## Tips for Best Results

### Photo Collection
- Take photos in similar lighting conditions
- Use consistent camera distance
- Include the dog's whole face in most shots
- Avoid extreme close-ups or distant shots

### Training
- Start with fewer steps (400-600) for initial testing
- If results are inconsistent, increase training steps
- If results are "overfitted" (too similar to training images), reduce steps
- Use validation prompts to check progress during training

### Quality Check
- After training, generate test images with various emotions
- Check for consistency across different prompts
- If quality is poor, retrain with more/better images

## Troubleshooting

### Issue: Model doesn't look like Butcher
- **Solution**: Add more training images (aim for 15-20)
- **Solution**: Ensure images clearly show Butcher's features
- **Solution**: Increase training steps to 1200-1500

### Issue: Images are blurry or low quality
- **Solution**: Start with higher resolution training images
- **Solution**: Use better quality source photos
- **Solution**: Reduce learning rate to 2e-6

### Issue: Model only generates training poses
- **Solution**: Add more variety to training images
- **Solution**: Reduce training steps to avoid overfitting
- **Solution**: Increase prior preservation weight

### Issue: Training is very slow
- **Solution**: Ensure GPU is being used (check with `nvidia-smi`)
- **Solution**: Reduce resolution to 512 in config
- **Solution**: Enable gradient checkpointing

## Instance Prompt Format

The instance prompt uses a unique identifier to represent Butcher:

- **Training prompt**: `"a photo of sks dog"`
- **Generation prompts**:
  - `"a photo of sks dog, happy expression"`
  - `"a photo of sks dog, french bulldog, excited"`
  - `"a photo of sks dog, portrait, professional photography"`

The token "sks" is a rare token that the model learns to associate with Butcher's appearance.

## Next Steps

1. ✅ Collect 10-20 photos of Butcher
2. ✅ Prepare training data using the script
3. ✅ Validate dataset quality
4. ✅ Review/edit training config
5. ✅ Run training
6. ✅ Generate test images
7. ✅ Integrate with video generation pipeline

## Resources

- [DreamBooth Paper](https://arxiv.org/abs/2208.12242)
- [Hugging Face DreamBooth Guide](https://huggingface.co/docs/diffusers/training/dreambooth)
- [Project Documentation](../../docs/)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review training logs in `logs/dreambooth/`
3. Validate your training data quality
4. Consult the main project README
