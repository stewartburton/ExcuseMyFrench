#!/usr/bin/env python3
"""
Generate images using Stable Diffusion with DreamBooth for characters.

This script uses Stable Diffusion to generate missing character images.
For Butcher, it uses a custom DreamBooth/LoRA model if available.
For Nutsy, it uses the base Stable Diffusion model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generates images using Stable Diffusion."""

    def __init__(self, use_gpu: bool = None):
        """
        Initialize the ImageGenerator.

        Args:
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
        """
        if use_gpu is None:
            use_gpu = os.getenv("USE_GPU", "true").lower() == "true"

        self.device = self._setup_device(use_gpu)

        # Model paths
        self.base_model = os.getenv(
            "STABLE_DIFFUSION_MODEL",
            "runwayml/stable-diffusion-v1-5"
        )
        self.dreambooth_path = Path(os.getenv(
            "DREAMBOOTH_MODEL_PATH",
            "models/dreambooth_butcher"
        ))
        self.lora_path = Path(os.getenv(
            "LORA_MODEL_PATH",
            "models/lora"
        ))

        # Image settings
        self.width = int(os.getenv("VIDEO_WIDTH", "1080"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "1920"))
        self.batch_size = int(os.getenv("BATCH_SIZE", "1"))

        # Initialize pipelines
        self.base_pipeline = None
        self.butcher_pipeline = None

        logger.info(f"Using device: {self.device}")
        logger.info(f"Image size: {self.width}x{self.height}")

    def _setup_device(self, use_gpu: bool) -> str:
        """
        Setup compute device.

        Args:
            use_gpu: Whether to attempt GPU usage

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if use_gpu and torch.cuda.is_available():
            gpu_id = int(os.getenv("GPU_DEVICE", "0"))
            device = f"cuda:{gpu_id}"
            logger.info(f"GPU available: {torch.cuda.get_device_name(gpu_id)}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.1f} GB")
        else:
            device = "cpu"
            if use_gpu:
                logger.warning("GPU requested but not available, using CPU")

        return device

    def _load_base_pipeline(self):
        """Load the base Stable Diffusion pipeline."""
        if self.base_pipeline is not None:
            return

        try:
            from diffusers import StableDiffusionPipeline

            logger.info(f"Loading base model: {self.base_model}")

            self.base_pipeline = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.base_pipeline = self.base_pipeline.to(self.device)

            # Enable memory optimization
            if "cuda" in self.device:
                self.base_pipeline.enable_attention_slicing()

            logger.info("Base pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load base pipeline: {e}")
            raise

    def _load_butcher_pipeline(self):
        """Load the DreamBooth pipeline for Butcher."""
        if self.butcher_pipeline is not None:
            return

        # Check if DreamBooth model exists
        if not self.dreambooth_path.exists():
            logger.warning(f"DreamBooth model not found at {self.dreambooth_path}")
            logger.warning("Using base model for Butcher (not recommended)")
            self._load_base_pipeline()
            self.butcher_pipeline = self.base_pipeline
            return

        try:
            from diffusers import StableDiffusionPipeline

            logger.info(f"Loading DreamBooth model: {self.dreambooth_path}")

            self.butcher_pipeline = StableDiffusionPipeline.from_pretrained(
                str(self.dreambooth_path),
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            self.butcher_pipeline = self.butcher_pipeline.to(self.device)

            # Enable memory optimization
            if "cuda" in self.device:
                self.butcher_pipeline.enable_attention_slicing()

            logger.info("Butcher pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Butcher pipeline: {e}")
            logger.warning("Falling back to base model")
            self._load_base_pipeline()
            self.butcher_pipeline = self.base_pipeline

    def _build_prompt(self, character: str, emotion: str) -> Dict[str, str]:
        """
        Build prompts for image generation.

        Args:
            character: Character name (Butcher or Nutsy)
            emotion: Emotion to convey

        Returns:
            Dictionary with 'prompt' and 'negative_prompt'
        """
        # Emotion descriptions
        emotion_prompts = {
            "happy": "smiling, joyful expression, cheerful",
            "sad": "sad expression, downcast, melancholy",
            "excited": "excited expression, energetic, enthusiastic",
            "sarcastic": "smirking, knowing look, raised eyebrow",
            "angry": "angry expression, furrowed brow, fierce",
            "confused": "confused expression, puzzled, questioning look",
            "surprised": "surprised expression, wide eyes, amazed",
            "neutral": "neutral expression, calm, relaxed"
        }

        emotion_desc = emotion_prompts.get(emotion.lower(), "neutral expression")

        if character == "Butcher":
            # DreamBooth instance prompt
            prompt = f"a photo of sks dog, French bulldog, {emotion_desc}, "
            prompt += "portrait, high quality, detailed, professional photography, "
            prompt += "clear face, front view, looking at camera"

        elif character == "Nutsy":
            prompt = f"a cute squirrel character, {emotion_desc}, "
            prompt += "cartoon style, anthropomorphic, expressive face, "
            prompt += "portrait, high quality, detailed, clear face, front view"

        else:
            raise ValueError(f"Unknown character: {character}")

        # Common negative prompt
        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, watermark, text, signature, "
            "multiple animals, duplicate, side view, back view"
        )

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt
        }

    def generate_image(
        self,
        character: str,
        emotion: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate a single image.

        Args:
            character: Character name
            emotion: Emotion to convey
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image
        """
        # Load appropriate pipeline
        if character == "Butcher":
            self._load_butcher_pipeline()
            pipeline = self.butcher_pipeline
        else:
            self._load_base_pipeline()
            pipeline = self.base_pipeline

        # Build prompts
        prompts = self._build_prompt(character, emotion)

        # Set seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        logger.info(f"Generating image: {character}/{emotion}")
        logger.debug(f"Prompt: {prompts['prompt']}")

        # Generate image
        try:
            result = pipeline(
                prompt=prompts['prompt'],
                negative_prompt=prompts['negative_prompt'],
                height=self.height,
                width=self.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            )

            image = result.images[0]
            logger.info("Image generated successfully")
            return image

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            raise

    def save_image(
        self,
        image: Image.Image,
        character: str,
        emotion: str,
        output_dir: str = "data/generated"
    ) -> str:
        """
        Save generated image to file.

        Args:
            image: PIL Image to save
            character: Character name
            emotion: Emotion
            output_dir: Directory to save to

        Returns:
            Path to saved file
        """
        from datetime import datetime

        output_path = Path(output_dir) / character.lower()
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{character.lower()}_{emotion.lower()}_{timestamp}.png"
        filepath = output_path / filename

        image.save(filepath)
        logger.info(f"Saved image to {filepath}")

        return str(filepath)

    def add_to_library(
        self,
        image_path: str,
        character: str,
        emotion: str,
        db_path: str = None
    ):
        """
        Add generated image to the image library database.

        Args:
            image_path: Path to the image file
            character: Character name
            emotion: Emotion
            db_path: Path to image library database
        """
        # Import here to avoid circular dependency
        from select_images import ImageSelector

        if db_path is None:
            db_path = os.getenv("IMAGE_LIBRARY_DB_PATH", "data/image_library.db")

        selector = ImageSelector(db_path=db_path)
        selector.add_image(
            character=character,
            emotion=emotion,
            file_path=image_path,
            source="generated_sd",
            quality_score=0.8  # Default quality for generated images
        )

    def generate_missing_images(
        self,
        missing_list: List[Dict[str, str]],
        output_dir: str = "data/generated",
        add_to_library: bool = True
    ) -> List[str]:
        """
        Generate all missing images from a list.

        Args:
            missing_list: List of dicts with 'character' and 'emotion'
            output_dir: Directory to save images
            add_to_library: Whether to add to image library database

        Returns:
            List of generated image paths
        """
        generated_paths = []

        # Remove duplicates
        unique_missing = []
        seen = set()
        for item in missing_list:
            key = (item['character'], item['emotion'])
            if key not in seen:
                seen.add(key)
                unique_missing.append(item)

        logger.info(f"Generating {len(unique_missing)} unique images")

        for i, item in enumerate(unique_missing, 1):
            character = item['character']
            emotion = item['emotion']

            try:
                logger.info(f"[{i}/{len(unique_missing)}] Generating {character}/{emotion}")

                image = self.generate_image(character, emotion)
                image_path = self.save_image(image, character, emotion, output_dir)
                generated_paths.append(image_path)

                if add_to_library:
                    self.add_to_library(image_path, character, emotion)

            except Exception as e:
                logger.error(f"Failed to generate {character}/{emotion}: {e}")
                continue

        logger.info(f"Successfully generated {len(generated_paths)} images")
        return generated_paths


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate character images using Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a single image
  python generate_images.py --character Butcher --emotion sarcastic

  # Generate from missing images list
  python generate_images.py --missing-json missing_images.json

  # Generate without adding to library
  python generate_images.py --character Nutsy --emotion excited --no-library

  # Use CPU instead of GPU
  python generate_images.py --character Butcher --emotion happy --cpu
        """
    )

    parser.add_argument(
        "--character",
        choices=["Butcher", "Nutsy"],
        help="Character to generate image for"
    )

    parser.add_argument(
        "--emotion",
        default="neutral",
        help="Emotion to convey (default: neutral)"
    )

    parser.add_argument(
        "--missing-json",
        help="JSON file with list of missing images to generate"
    )

    parser.add_argument(
        "--output-dir",
        default="data/generated",
        help="Directory to save generated images (default: data/generated)"
    )

    parser.add_argument(
        "--no-library",
        action="store_true",
        help="Don't add generated images to library database"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (default: 50)"
    )

    parser.add_argument(
        "--guidance",
        type=float,
        default=7.5,
        help="Guidance scale (default: 7.5)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize generator
    try:
        generator = ImageGenerator(use_gpu=not args.cpu)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        sys.exit(1)

    if args.missing_json:
        # Generate from missing list
        try:
            with open(args.missing_json, 'r') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'missing' in data:
                missing_list = data['missing']
            elif isinstance(data, list):
                missing_list = data
            else:
                logger.error("Invalid JSON format")
                sys.exit(1)

            generated = generator.generate_missing_images(
                missing_list,
                output_dir=args.output_dir,
                add_to_library=not args.no_library
            )

            print(f"\nGenerated {len(generated)} images:")
            for path in generated:
                print(f"  - {path}")

        except Exception as e:
            logger.error(f"Failed to process missing list: {e}")
            sys.exit(1)

    elif args.character:
        # Generate single image
        try:
            image = generator.generate_image(
                character=args.character,
                emotion=args.emotion,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed
            )

            image_path = generator.save_image(
                image,
                args.character,
                args.emotion,
                args.output_dir
            )

            if not args.no_library:
                generator.add_to_library(
                    image_path,
                    args.character,
                    args.emotion
                )

            print(f"\nGenerated image: {image_path}")

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
