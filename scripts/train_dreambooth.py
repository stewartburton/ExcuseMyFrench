#!/usr/bin/env python3
"""
Train DreamBooth model for Butcher character using Stable Diffusion.

This script fine-tunes a Stable Diffusion model on Butcher's images
to enable consistent character generation across different emotions and poses.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DreamBoothTrainer:
    """Trainer for DreamBooth fine-tuning."""

    def __init__(
        self,
        character: str = "Butcher",
        instance_prompt: str = None,
        class_prompt: str = None
    ):
        """
        Initialize the DreamBooth trainer.

        Args:
            character: Character name to train
            instance_prompt: Prompt for training images (e.g., "a photo of sks dog")
            class_prompt: Class prompt for regularization (e.g., "a photo of dog")
        """
        self.character = character

        # Default prompts for Butcher
        if instance_prompt is None:
            instance_prompt = "a photo of sks dog"
        if class_prompt is None:
            class_prompt = "a photo of a dog"

        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        # Model settings
        self.base_model = os.getenv(
            "STABLE_DIFFUSION_MODEL",
            "runwayml/stable-diffusion-v1-5"
        )

        # Training settings from environment
        self.max_train_steps = int(os.getenv("DREAMBOOTH_TRAIN_STEPS", "800"))
        self.learning_rate = float(os.getenv("DREAMBOOTH_LEARNING_RATE", "5e-6"))
        self.train_batch_size = int(os.getenv("DREAMBOOTH_BATCH_SIZE", "1"))
        self.gradient_accumulation_steps = int(os.getenv("DREAMBOOTH_GRADIENT_ACCUM", "1"))
        self.resolution = int(os.getenv("DREAMBOOTH_RESOLUTION", "512"))

        logger.info(f"Training DreamBooth model for {character}")
        logger.info(f"Instance prompt: {instance_prompt}")
        logger.info(f"Max training steps: {self.max_train_steps}")

    def prepare_training_data(
        self,
        instance_data_dir: str,
        class_data_dir: Optional[str] = None
    ):
        """
        Prepare training data directory structure.

        Args:
            instance_data_dir: Directory containing character images
            class_data_dir: Optional directory for class images (regularization)
        """
        instance_path = Path(instance_data_dir)

        if not instance_path.exists():
            raise FileNotFoundError(f"Instance data directory not found: {instance_data_dir}")

        # Count training images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = [
            f for f in instance_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(images)} training images in {instance_data_dir}")

        if len(images) < 5:
            logger.warning("Less than 5 training images found - consider adding more for better results")

        return len(images)

    def train_with_diffusers(
        self,
        instance_data_dir: str,
        output_dir: str,
        class_data_dir: Optional[str] = None,
        use_prior_preservation: bool = True,
        num_class_images: int = 200,
        use_8bit_adam: bool = False,
        gradient_checkpointing: bool = True,
        mixed_precision: str = "fp16"
    ):
        """
        Train DreamBooth model using Hugging Face Diffusers.

        Args:
            instance_data_dir: Directory with character training images
            output_dir: Directory to save trained model
            class_data_dir: Directory for class images (optional)
            use_prior_preservation: Use prior preservation loss
            num_class_images: Number of class images to generate
            use_8bit_adam: Use 8-bit Adam optimizer (saves VRAM)
            gradient_checkpointing: Enable gradient checkpointing (saves VRAM)
            mixed_precision: Mixed precision training ("no", "fp16", "bf16")
        """
        try:
            import torch
            from diffusers import StableDiffusionPipeline, DDPMScheduler
            from accelerate import Accelerator
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.info("Install with: pip install diffusers accelerate transformers")
            sys.exit(1)

        # Check GPU
        if not torch.cuda.is_available():
            logger.warning("No GPU detected - training will be very slow")
            logger.warning("Consider using a GPU with at least 16GB VRAM")

        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("DREAMBOOTH TRAINING CONFIGURATION")
        logger.info("=" * 80)
        logger.info(f"Base model: {self.base_model}")
        logger.info(f"Instance data: {instance_data_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Instance prompt: {self.instance_prompt}")
        logger.info(f"Class prompt: {self.class_prompt}")
        logger.info(f"Training steps: {self.max_train_steps}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Resolution: {self.resolution}")
        logger.info(f"Mixed precision: {mixed_precision}")
        logger.info(f"Gradient checkpointing: {gradient_checkpointing}")
        logger.info(f"Prior preservation: {use_prior_preservation}")
        logger.info("=" * 80)

        # Import training script functionality
        try:
            logger.info("Starting DreamBooth training...")
            logger.info("This may take 1-4 hours depending on GPU and settings")

            # Use Hugging Face Accelerate launch
            cmd = self._build_training_command(
                instance_data_dir=instance_data_dir,
                output_dir=output_dir,
                class_data_dir=class_data_dir,
                use_prior_preservation=use_prior_preservation,
                num_class_images=num_class_images,
                use_8bit_adam=use_8bit_adam,
                gradient_checkpointing=gradient_checkpointing,
                mixed_precision=mixed_precision
            )

            logger.info(f"Training command: {' '.join(cmd)}")

            # Run training
            import subprocess
            result = subprocess.run(cmd, check=True)

            logger.info("✓ Training completed successfully!")
            logger.info(f"Model saved to: {output_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during training: {e}")
            raise

    def _build_training_command(
        self,
        instance_data_dir: str,
        output_dir: str,
        class_data_dir: Optional[str],
        use_prior_preservation: bool,
        num_class_images: int,
        use_8bit_adam: bool,
        gradient_checkpointing: bool,
        mixed_precision: str
    ) -> list:
        """Build the accelerate launch command for training."""
        cmd = [
            "accelerate", "launch",
            "train_dreambooth.py"  # Assumes Hugging Face diffusers example script
        ]

        # Add required arguments
        cmd.extend([
            "--pretrained_model_name_or_path", self.base_model,
            "--instance_data_dir", instance_data_dir,
            "--output_dir", output_dir,
            "--instance_prompt", self.instance_prompt,
            "--resolution", str(self.resolution),
            "--train_batch_size", str(self.train_batch_size),
            "--gradient_accumulation_steps", str(self.gradient_accumulation_steps),
            "--learning_rate", str(self.learning_rate),
            "--lr_scheduler", "constant",
            "--lr_warmup_steps", "0",
            "--max_train_steps", str(self.max_train_steps),
            "--mixed_precision", mixed_precision
        ])

        # Prior preservation
        if use_prior_preservation and class_data_dir:
            cmd.extend([
                "--with_prior_preservation",
                "--prior_loss_weight", "1.0",
                "--class_data_dir", class_data_dir,
                "--class_prompt", self.class_prompt,
                "--num_class_images", str(num_class_images)
            ])

        # Memory optimization
        if gradient_checkpointing:
            cmd.append("--gradient_checkpointing")

        if use_8bit_adam:
            cmd.append("--use_8bit_adam")

        return cmd

    def test_model(
        self,
        model_path: str,
        output_dir: str = "data/test_outputs",
        test_prompts: list = None
    ):
        """
        Test trained DreamBooth model by generating sample images.

        Args:
            model_path: Path to trained model
            output_dir: Directory to save test images
            test_prompts: List of prompts to test (uses defaults if None)
        """
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            sys.exit(1)

        if test_prompts is None:
            test_prompts = [
                "a photo of sks dog, happy expression, professional photography",
                "a photo of sks dog, sarcastic expression, high quality",
                "a photo of sks dog, excited expression, portrait",
                "a photo of sks dog, neutral expression, front view"
            ]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading model from {model_path}")

        # Load trained model
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None
        )

        if torch.cuda.is_available():
            pipeline = pipeline.to("cuda")

        logger.info("Generating test images...")

        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"[{i}/{len(test_prompts)}] {prompt}")

            image = pipeline(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=7.5
            ).images[0]

            # Save image
            filename = f"test_{i:02d}.png"
            filepath = output_path / filename
            image.save(filepath)

            logger.info(f"Saved: {filepath}")

        logger.info(f"✓ Test images saved to: {output_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Train DreamBooth model for character consistency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train Butcher model with default settings
  python train_dreambooth.py \\
    --instance-data data/butcher_training \\
    --output models/dreambooth_butcher

  # Train with prior preservation and class images
  python train_dreambooth.py \\
    --instance-data data/butcher_training \\
    --class-data data/class_dogs \\
    --output models/dreambooth_butcher \\
    --prior-preservation

  # Train with custom settings (more steps, higher resolution)
  python train_dreambooth.py \\
    --instance-data data/butcher_training \\
    --output models/dreambooth_butcher \\
    --steps 1200 \\
    --resolution 768 \\
    --learning-rate 2e-6

  # Test trained model
  python train_dreambooth.py \\
    --test models/dreambooth_butcher \\
    --test-output data/test_outputs
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--instance-data",
        help="Directory containing character training images"
    )
    mode_group.add_argument(
        "--test",
        help="Test a trained model (provide model path)"
    )

    # Training arguments
    parser.add_argument(
        "--output",
        help="Output directory for trained model"
    )

    parser.add_argument(
        "--class-data",
        help="Directory for class images (prior preservation)"
    )

    parser.add_argument(
        "--prior-preservation",
        action="store_true",
        help="Use prior preservation loss"
    )

    parser.add_argument(
        "--instance-prompt",
        default="a photo of sks dog",
        help="Instance prompt (default: 'a photo of sks dog')"
    )

    parser.add_argument(
        "--class-prompt",
        default="a photo of a dog",
        help="Class prompt (default: 'a photo of a dog')"
    )

    parser.add_argument(
        "--steps",
        type=int,
        help="Max training steps (default: from .env or 800)"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (default: from .env or 5e-6)"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        choices=[512, 768, 1024],
        help="Training resolution (default: from .env or 512)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (default: 1)"
    )

    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="fp16",
        help="Mixed precision training (default: fp16)"
    )

    parser.add_argument(
        "--use-8bit-adam",
        action="store_true",
        help="Use 8-bit Adam optimizer (saves VRAM)"
    )

    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing"
    )

    # Test arguments
    parser.add_argument(
        "--test-output",
        default="data/test_outputs",
        help="Directory for test outputs (default: data/test_outputs)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Test mode
    if args.test:
        trainer = DreamBoothTrainer()
        trainer.test_model(
            model_path=args.test,
            output_dir=args.test_output
        )
        return

    # Training mode
    if not args.output:
        parser.error("--output is required for training")

    # Override environment settings with command-line args
    if args.steps:
        os.environ["DREAMBOOTH_TRAIN_STEPS"] = str(args.steps)
    if args.learning_rate:
        os.environ["DREAMBOOTH_LEARNING_RATE"] = str(args.learning_rate)
    if args.resolution:
        os.environ["DREAMBOOTH_RESOLUTION"] = str(args.resolution)
    if args.batch_size:
        os.environ["DREAMBOOTH_BATCH_SIZE"] = str(args.batch_size)

    # Initialize trainer
    trainer = DreamBoothTrainer(
        character="Butcher",
        instance_prompt=args.instance_prompt,
        class_prompt=args.class_prompt
    )

    # Prepare training data
    try:
        num_images = trainer.prepare_training_data(
            instance_data_dir=args.instance_data,
            class_data_dir=args.class_data
        )

        if num_images == 0:
            logger.error("No training images found!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        sys.exit(1)

    # Confirm training
    print("\n" + "=" * 80)
    print("DREAMBOOTH TRAINING")
    print("=" * 80)
    print(f"Training images: {num_images}")
    print(f"Training steps: {trainer.max_train_steps}")
    print(f"Estimated time: {trainer.max_train_steps * 3 // 60}-{trainer.max_train_steps * 5 // 60} minutes")
    print("=" * 80)

    response = input("\nStart training? [y/N]: ")
    if response.lower() != 'y':
        print("Training cancelled")
        return

    # Train model
    try:
        trainer.train_with_diffusers(
            instance_data_dir=args.instance_data,
            output_dir=args.output,
            class_data_dir=args.class_data,
            use_prior_preservation=args.prior_preservation,
            use_8bit_adam=args.use_8bit_adam,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            mixed_precision=args.mixed_precision
        )

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Model saved to: {args.output}")
        print("\nNext steps:")
        print(f"  1. Test model: python train_dreambooth.py --test {args.output}")
        print(f"  2. Update .env: DREAMBOOTH_MODEL_PATH={args.output}")
        print("  3. Use in pipeline: python scripts/generate_images.py --character Butcher")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
