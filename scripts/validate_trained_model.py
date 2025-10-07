#!/usr/bin/env python3
"""
Post-Training Model Validation Script

Automatically validates the trained DreamBooth model by:
1. Detecting latest trained model checkpoint
2. Generating test images with various emotions
3. Comparing quality with training images
4. Creating a quality report with recommendations

Usage:
    python scripts/validate_trained_model.py
    python scripts/validate_trained_model.py --model models/dreambooth_butcher/
    python scripts/validate_trained_model.py --output reports/model_validation.md
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def find_latest_checkpoint(model_dir: Path) -> Path:
    """Find the latest checkpoint directory or final model."""
    model_dir = Path(model_dir)

    # Check if there's a final model (no checkpoint prefix)
    if (model_dir / "model_index.json").exists():
        return model_dir

    # Find all checkpoint directories
    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    # Sort by checkpoint number and get the latest
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return checkpoints[-1]


def load_model(model_path: Path):
    """Load the trained DreamBooth model."""
    print(f"Loading model from {model_path}...")

    try:
        pipeline = StableDiffusionPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )

        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = pipeline.to(device)

        print(f"Model loaded successfully on {device}")
        return pipeline

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def generate_test_image(pipeline, emotion: str, seed: int = 42, output_dir: Path = None) -> Path:
    """Generate a test image with specified emotion."""
    prompts = {
        "happy": "a photo of sks dog, happy expression, smiling, bright eyes, professional photography, studio lighting",
        "sarcastic": "a photo of sks dog, sarcastic smirk, side-eye glance, skeptical expression, professional photography",
        "grumpy": "a photo of sks dog, grumpy expression, furrowed brow, annoyed look, professional photography",
        "neutral": "a photo of sks dog, neutral expression, looking at camera, professional photography, studio lighting",
        "excited": "a photo of sks dog, excited expression, ears perked up, energetic pose, professional photography",
        "confused": "a photo of sks dog, confused expression, head tilted, questioning look, professional photography"
    }

    prompt = prompts.get(emotion, prompts["neutral"])
    negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, extra limbs"

    print(f"Generating {emotion} image...")
    print(f"Prompt: {prompt}")

    # Set random seed for reproducibility
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    # Generate image
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    # Save image
    if output_dir:
        output_path = output_dir / f"test_{emotion}_{seed}.png"
        image.save(output_path)
        print(f"Saved to {output_path}")
        return output_path

    return image


def analyze_training_data(training_dir: Path) -> dict:
    """Analyze the training data used to train the model."""
    training_images = list((training_dir / "images").glob("*.jpg")) + \
                     list((training_dir / "images").glob("*.png"))

    total_images = len(training_images)

    # Get image dimensions (sample first image)
    if training_images:
        sample_img = Image.open(training_images[0])
        typical_size = sample_img.size
    else:
        typical_size = (0, 0)

    return {
        "total_images": total_images,
        "typical_size": typical_size,
        "image_paths": [str(p) for p in training_images[:5]]  # First 5 for reference
    }


def create_validation_report(
    model_path: Path,
    test_images: dict,
    training_data: dict,
    output_file: Path
):
    """Create a markdown validation report."""
    report = []
    report.append(f"# DreamBooth Model Validation Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Model:** `{model_path}`\n")
    report.append("\n---\n")

    # Model Information
    report.append("## Model Information\n")
    report.append(f"- **Model Path:** `{model_path}`\n")
    report.append(f"- **Model Type:** DreamBooth fine-tuned Stable Diffusion\n")
    report.append(f"- **Training Images:** {training_data['total_images']} images\n")
    report.append(f"- **Typical Training Image Size:** {training_data['typical_size']}\n")
    report.append("\n")

    # Training Data Sample
    report.append("### Training Data Sample\n")
    for i, img_path in enumerate(training_data['image_paths'], 1):
        report.append(f"{i}. `{Path(img_path).name}`\n")
    report.append("\n")

    # Test Results
    report.append("## Validation Test Results\n")
    report.append(f"Generated {len(test_images)} test images with various emotions:\n\n")

    for emotion, img_path in test_images.items():
        report.append(f"### {emotion.capitalize()} Expression\n")
        report.append(f"- **Image:** `{img_path.name}`\n")
        report.append(f"- **Full Path:** `{img_path}`\n")
        report.append(f"![{emotion}]({img_path})\n")
        report.append("\n")

    # Quality Checklist
    report.append("## Manual Quality Review Checklist\n")
    report.append("Review the generated images and check:\n\n")
    report.append("- [ ] **Character Consistency:** All images show the same character (Butcher)\n")
    report.append("- [ ] **Breed Accuracy:** Character appears as French Bulldog\n")
    report.append("- [ ] **Facial Features:** Distinctive features match training images\n")
    report.append("- [ ] **Emotion Clarity:** Each emotion is clearly distinguishable\n")
    report.append("- [ ] **Image Quality:** No artifacts, distortions, or deformities\n")
    report.append("- [ ] **Background:** Clean backgrounds, professional look\n")
    report.append("- [ ] **Lighting:** Consistent, good lighting across images\n")
    report.append("- [ ] **Resolution:** Images are sharp and high quality\n")
    report.append("\n")

    # Recommendations
    report.append("## Recommendations\n")
    report.append("### If Quality is Good ✅\n")
    report.append("- ✅ Model is ready for production use\n")
    report.append("- ✅ Proceed to generate character images for videos\n")
    report.append("- ✅ Test with `scripts/generate_character_image.py`\n")
    report.append("\n")

    report.append("### If Quality is Poor ❌\n")
    report.append("**Common Issues and Solutions:**\n\n")
    report.append("1. **Inconsistent Character Appearance**\n")
    report.append("   - **Cause:** Insufficient or low-quality training data\n")
    report.append("   - **Solution:** Add more training images (15-25), ensure diversity\n")
    report.append("   - **Action:** Collect more images, retrain model\n\n")

    report.append("2. **Generic/Stock Photos**\n")
    report.append("   - **Cause:** Model not properly learning the specific character\n")
    report.append("   - **Solution:** Increase training steps (try 1200 instead of 800)\n")
    report.append("   - **Action:** Retrain with longer duration\n\n")

    report.append("3. **Blurry or Distorted Images**\n")
    report.append("   - **Cause:** Training data had varying quality or sizes\n")
    report.append("   - **Solution:** Ensure all training images are high-quality, similar resolution\n")
    report.append("   - **Action:** Clean up training dataset, retrain\n\n")

    report.append("4. **Wrong Breed/Species**\n")
    report.append("   - **Cause:** Class images (dog general images) overwhelmed the instance images\n")
    report.append("   - **Solution:** Check class_weight in config (should be 1.0 for prior_preservation_weight)\n")
    report.append("   - **Action:** Verify config, possibly retrain\n\n")

    report.append("5. **Emotions Not Clear**\n")
    report.append("   - **Cause:** Test prompts may need adjustment\n")
    report.append("   - **Solution:** Refine prompts with more specific emotion descriptors\n")
    report.append("   - **Action:** Try generating with adjusted prompts first\n\n")

    # Next Steps
    report.append("## Next Steps\n")
    report.append("1. **Review all generated test images** in `{}`\n".format(test_images[list(test_images.keys())[0]].parent))
    report.append("2. **Complete the quality checklist** above\n")
    report.append("3. **If quality is good:**\n")
    report.append("   - Generate production character images: `python scripts/generate_character_image.py --character butcher --emotion [EMOTION]`\n")
    report.append("   - Test in full video pipeline: `make run-pipeline`\n")
    report.append("4. **If quality needs improvement:**\n")
    report.append("   - Follow recommendations above\n")
    report.append("   - Retrain model with adjustments\n")
    report.append("   - Run validation again\n")
    report.append("\n")

    # Save report
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"\nValidation report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Validate trained DreamBooth model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/dreambooth_butcher",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/validation",
        help="Directory to save test images"
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/model_validation.md",
        help="Path to save validation report"
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=["happy", "sarcastic", "grumpy", "neutral"],
        help="Emotions to test"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip image generation, only create report from existing images"
    )

    args = parser.parse_args()

    # Convert paths
    model_dir = Path(args.model)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report)

    print("=" * 70)
    print("DreamBooth Model Validation")
    print("=" * 70)

    # Check if model directory exists
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("\nTip: Make sure DreamBooth training has completed.")
        print("     Check models/dreambooth_butcher/ for the trained model.")
        sys.exit(1)

    # Find latest checkpoint
    try:
        model_path = find_latest_checkpoint(model_dir)
        print(f"Using model: {model_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Analyze training data
    training_dir = Path("training/butcher")
    if training_dir.exists():
        training_data = analyze_training_data(training_dir)
        print(f"Training data: {training_data['total_images']} images")
    else:
        print(f"Warning: Training directory not found at {training_dir}")
        training_data = {"total_images": 0, "typical_size": (0, 0), "image_paths": []}

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test images
    test_images = {}

    if not args.skip_generation:
        # Load model
        try:
            pipeline = load_model(model_path)
        except Exception as e:
            print(f"\nFailed to load model: {e}")
            print("\nThis could mean:")
            print("1. Training is not yet complete")
            print("2. Model files are corrupted")
            print("3. Insufficient GPU memory")
            sys.exit(1)

        # Generate test images for each emotion
        print(f"\nGenerating test images for emotions: {', '.join(args.emotions)}")
        for emotion in args.emotions:
            try:
                img_path = generate_test_image(pipeline, emotion, args.seed, output_dir)
                test_images[emotion] = img_path
            except Exception as e:
                print(f"Error generating {emotion} image: {e}")
                continue

        print(f"\nGenerated {len(test_images)} test images in {output_dir}")

    else:
        # Find existing test images
        print(f"Looking for existing test images in {output_dir}...")
        for emotion in args.emotions:
            img_files = list(output_dir.glob(f"test_{emotion}_*.png"))
            if img_files:
                test_images[emotion] = img_files[0]
                print(f"Found {emotion}: {img_files[0].name}")

    if not test_images:
        print("\nError: No test images generated or found.")
        sys.exit(1)

    # Create validation report
    print(f"\nCreating validation report...")
    create_validation_report(model_path, test_images, training_data, report_path)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Test Images: {output_dir}")
    print(f"Report: {report_path}")
    print("\nNext steps:")
    print(f"1. Open the report: {report_path}")
    print(f"2. Review the generated images in: {output_dir}")
    print(f"3. Complete the quality checklist")
    print("4. Follow recommendations based on quality assessment")
    print("=" * 70)


if __name__ == "__main__":
    main()
