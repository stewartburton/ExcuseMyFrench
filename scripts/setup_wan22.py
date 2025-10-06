#!/usr/bin/env python3
"""
Download and setup Wan 2.2 video generation models.

This script handles downloading Wan 2.2 models from Hugging Face
and configuring them for the Excuse My French pipeline.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

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


def check_requirements():
    """Check if required packages are installed."""
    required = ["diffusers", "transformers", "accelerate", "torch"]
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.info("Install with: pip install " + " ".join(missing))
        return False

    return True


def check_hf_token():
    """Check if Hugging Face token is configured."""
    token = os.getenv("HF_TOKEN")

    if not token or token == "hf_your-huggingface-token-here":
        logger.error("HF_TOKEN not configured in .env file")
        logger.info("Get your token at: https://huggingface.co/settings/tokens")
        return False

    return True


def get_disk_space(path: Path) -> float:
    """
    Get available disk space in GB.

    Args:
        path: Path to check

    Returns:
        Available space in GB
    """
    import shutil
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def download_wan_model(model_id: str, save_path: Path, use_fp16: bool = True):
    """
    Download Wan 2.2 model from Hugging Face.

    Args:
        model_id: Hugging Face model ID
        save_path: Path to save the model
        use_fp16: Use float16 precision (recommended)
    """
    try:
        from diffusers import WanPipeline
        import torch

        logger.info(f"Downloading {model_id}...")
        logger.info(f"Save location: {save_path}")

        # Download model
        pipeline = WanPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
            variant="fp16" if use_fp16 else None,
            use_safetensors=True
        )

        # Save model
        save_path.mkdir(parents=True, exist_ok=True)
        pipeline.save_pretrained(save_path)

        logger.info(f"Model saved successfully to {save_path}")

        # Calculate model size
        total_size = sum(
            f.stat().st_size
            for f in save_path.rglob('*')
            if f.is_file()
        ) / (1024 ** 3)

        logger.info(f"Model size: {total_size:.2f} GB")

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def setup_wan_models(model_size: str = "5B", models_to_download: list = None):
    """
    Set up Wan 2.2 models.

    Args:
        model_size: Model size ('5B' or '14B')
        models_to_download: List of model types to download ('i2v', 't2v', 's2v')
    """
    # Default models
    if models_to_download is None:
        models_to_download = ["i2v"]  # Image-to-Video is most useful for our pipeline

    # Base path
    wan_path = Path(os.getenv("WAN_MODEL_PATH", "models/wan2.2"))

    # Check disk space
    required_space = 40 if model_size == "5B" else 80
    available_space = get_disk_space(wan_path.parent if wan_path.exists() else Path.cwd())

    logger.info(f"Available disk space: {available_space:.2f} GB")
    logger.info(f"Required disk space: ~{required_space} GB")

    if available_space < required_space:
        logger.error(f"Insufficient disk space! Need at least {required_space} GB")
        return False

    # Model mappings
    model_ids = {
        "5B": {
            "i2v": "Wan-AI/Wan2.2-I2V-5B",
            "t2v": "Wan-AI/Wan2.2-T2V-5B",
            "s2v": "Wan-AI/Wan2.2-S2V-5B"
        },
        "14B": {
            "i2v": "Wan-AI/Wan2.2-I2V-A14B",
            "t2v": "Wan-AI/Wan2.2-T2V-A14B",
            "s2v": "Wan-AI/Wan2.2-S2V-14B"
        }
    }

    # Download requested models
    for model_type in models_to_download:
        if model_type not in model_ids[model_size]:
            logger.warning(f"Unknown model type: {model_type}, skipping")
            continue

        model_id = model_ids[model_size][model_type]
        save_path = wan_path / f"{model_type}-{model_size.lower()}"

        # Skip if already exists
        if save_path.exists() and (save_path / "model_index.json").exists():
            logger.info(f"Model already exists: {save_path}")
            continue

        try:
            download_wan_model(model_id, save_path)
        except Exception as e:
            logger.error(f"Failed to download {model_type}: {e}")
            return False

    logger.info("✓ Wan 2.2 setup complete!")
    return True


def verify_installation(model_size: str = "5B"):
    """
    Verify Wan 2.2 installation by loading a model.

    Args:
        model_size: Model size to verify
    """
    wan_path = Path(os.getenv("WAN_MODEL_PATH", "models/wan2.2"))
    model_path = wan_path / f"i2v-{model_size.lower()}"

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return False

    try:
        from diffusers import WanPipeline
        import torch

        logger.info("Loading model for verification...")

        pipeline = WanPipeline.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16
        )

        # Check if GPU is available
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

            # Recommend model based on VRAM
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 16 and model_size == "14B":
                logger.warning("14B model may not fit in your GPU memory!")
                logger.warning("Consider using 5B model instead")
            elif vram_gb >= 16:
                logger.info("✓ Sufficient VRAM for model")
        else:
            logger.warning("No GPU detected - inference will be very slow")

        logger.info("✓ Model loaded successfully!")
        return True

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download and setup Wan 2.2 video generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 5B Image-to-Video model (recommended)
  python setup_wan22.py --model-size 5B --models i2v

  # Download all 5B models
  python setup_wan22.py --model-size 5B --models i2v t2v s2v

  # Download 14B model (requires 24GB+ VRAM)
  python setup_wan22.py --model-size 14B --models i2v

  # Verify existing installation
  python setup_wan22.py --verify
        """
    )

    parser.add_argument(
        "--model-size",
        choices=["5B", "14B"],
        default="5B",
        help="Model size to download (default: 5B)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["i2v", "t2v", "s2v"],
        default=["i2v"],
        help="Model types to download (default: i2v)"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing installation"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check requirements
    logger.info("Checking requirements...")
    if not check_requirements():
        logger.error("Please install required packages first")
        logger.info("Run: pip install diffusers transformers accelerate torch")
        sys.exit(1)

    if not check_hf_token():
        logger.error("Please configure HF_TOKEN in config/.env")
        sys.exit(1)

    logger.info("✓ Requirements check passed")

    # Verify mode
    if args.verify:
        logger.info("Verifying installation...")
        if verify_installation(args.model_size):
            print("\n✓ Wan 2.2 is properly installed and ready to use!\n")
        else:
            print("\n✗ Verification failed - please run setup again\n")
            sys.exit(1)
        return

    # Setup mode
    print("\n" + "=" * 80)
    print("WAN 2.2 MODEL SETUP")
    print("=" * 80)
    print(f"Model size: {args.model_size}")
    print(f"Models to download: {', '.join(args.models)}")
    print("=" * 80 + "\n")

    # Confirm download
    if args.model_size == "14B":
        print("WARNING: 14B models require:")
        print("  - 24GB+ GPU VRAM (A100/H100 recommended)")
        print("  - ~80GB disk space")
        print("  - Long download time (1-2 hours depending on connection)")
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("Setup cancelled")
            return

    # Download models
    try:
        if setup_wan_models(args.model_size, args.models):
            # Verify installation
            if verify_installation(args.model_size):
                print("\n" + "=" * 80)
                print("SETUP COMPLETE!")
                print("=" * 80)
                print(f"Models installed to: {Path(os.getenv('WAN_MODEL_PATH', 'models/wan2.2'))}")
                print("\nNext steps:")
                print("  1. Update config/.env with WAN_MODEL_SIZE=" + args.model_size)
                print("  2. See docs/ANIMATION_SETUP.md for usage examples")
                print("  3. Test with: python scripts/test_wan22.py")
                print("=" * 80 + "\n")
        else:
            logger.error("Setup failed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
