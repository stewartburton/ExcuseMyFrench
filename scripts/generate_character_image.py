#!/usr/bin/env python3
"""
Character Image Generator using Trained DreamBooth Models

This script generates character images using custom-trained DreamBooth models.
It provides fine-grained control over emotions, poses, and quality.

Features:
- Use trained DreamBooth models for consistent characters
- Generate images with different emotions and poses
- Batch generation capability
- Quality filtering and validation
- Automatic retry with different seeds
- Integration with image library database

Usage:
    python scripts/generate_character_image.py --character butcher --emotion happy --count 5
    python scripts/generate_character_image.py --batch emotions.json --output data/generated
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from tqdm import tqdm
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


class CharacterImageGenerator:
    """Generates character images using DreamBooth models."""

    # Emotion to prompt mapping
    EMOTION_PROMPTS = {
        "happy": "happy, joyful, smiling, cheerful expression",
        "sad": "sad, melancholy, downcast, sorrowful expression",
        "excited": "excited, energetic, enthusiastic, animated expression",
        "sarcastic": "sarcastic, smirking, knowing look, raised eyebrow",
        "angry": "angry, fierce, intense, furrowed brow",
        "confused": "confused, puzzled, questioning, uncertain expression",
        "surprised": "surprised, shocked, amazed, wide eyes",
        "neutral": "neutral, calm, relaxed, natural expression",
        "playful": "playful, mischievous, fun, energetic expression",
        "grumpy": "grumpy, annoyed, irritated, displeased expression",
        "sleepy": "sleepy, tired, drowsy, relaxed expression",
        "thinking": "thinking, contemplative, thoughtful, focused expression",
    }

    # Pose/angle descriptions
    POSE_PROMPTS = {
        "front": "front view, facing camera, centered",
        "side": "side profile, side view",
        "slight_angle": "slightly turned, three-quarter view",
        "close_up": "close up, portrait shot",
        "full_body": "full body shot, complete view",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = None,
        safety_checker: bool = False,
    ):
        """
        Initialize the character image generator.

        Args:
            model_path: Path to DreamBooth model (None for default from env)
            use_gpu: Whether to use GPU (auto-detect if None)
            safety_checker: Whether to use safety checker
        """
        # Determine model path
        if model_path is None:
            model_path = os.getenv("DREAMBOOTH_MODEL_PATH", "models/dreambooth_butcher")

        self.model_path = Path(model_path)

        # Check if model exists
        if not self.model_path.exists():
            logger.warning(f"DreamBooth model not found at {self.model_path}")
            logger.warning("Falling back to base Stable Diffusion model")
            self.model_path = os.getenv("STABLE_DIFFUSION_MODEL", "runwayml/stable-diffusion-v1-5")

        # Setup device
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

        # Image settings
        self.width = int(os.getenv("VIDEO_WIDTH", "1080"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "1920"))

        # Pipeline
        self.pipeline = None
        self.safety_checker = safety_checker

        # Load model info if available
        self.model_info = self._load_model_info()

        logger.info(f"Character generator initialized")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Device: {self.device}")

    def _load_model_info(self) -> Dict:
        """Load model training info if available."""
        info_path = self.model_path / "model_info.json" if self.model_path.exists() else None

        if info_path and info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)

        return {
            "instance_prompt": "a photo of sks dog",
            "character": "butcher",
        }

    def load_pipeline(self):
        """Load the generation pipeline."""
        if self.pipeline is not None:
            return

        logger.info(f"Loading pipeline from {self.model_path}")

        try:
            # Load pipeline
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None if not self.safety_checker else "default",
            )

            # Use faster scheduler
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config
            )

            # Move to device
            self.pipeline = self.pipeline.to(self.device)

            # Enable optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                # Try to enable xformers if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers memory efficient attention enabled")
                except Exception:
                    pass

            logger.info("Pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def build_prompt(
        self,
        emotion: str = "neutral",
        pose: str = "front",
        additional_details: Optional[str] = None,
        character: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Build generation prompts.

        Args:
            emotion: Emotion to convey
            pose: Pose/angle
            additional_details: Additional prompt details
            character: Character name (uses model default if None)

        Returns:
            Tuple of (prompt, negative_prompt)
        """
        # Get character and instance token from model info
        if character is None:
            character = self.model_info.get("character", "butcher")

        instance_prompt = self.model_info.get("instance_prompt", "a photo of sks dog")

        # Build positive prompt
        prompt_parts = [instance_prompt]

        # Add emotion
        emotion_desc = self.EMOTION_PROMPTS.get(emotion.lower(), self.EMOTION_PROMPTS["neutral"])
        prompt_parts.append(emotion_desc)

        # Add pose
        pose_desc = self.POSE_PROMPTS.get(pose.lower(), self.POSE_PROMPTS["front"])
        prompt_parts.append(pose_desc)

        # Add quality modifiers
        prompt_parts.append("high quality, detailed, professional photography")

        # Add additional details
        if additional_details:
            prompt_parts.append(additional_details)

        prompt = ", ".join(prompt_parts)

        # Build negative prompt
        negative_prompt = (
            "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, watermark, text, signature, logo, "
            "multiple animals, duplicate, side view, back view, "
            "cartoon, drawing, illustration, painting"
        )

        return prompt, negative_prompt

    def generate(
        self,
        emotion: str = "neutral",
        pose: str = "front",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        additional_details: Optional[str] = None,
        character: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a character image.

        Args:
            emotion: Emotion to convey
            pose: Pose/angle
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            seed: Random seed
            additional_details: Additional prompt details
            character: Character name

        Returns:
            Generated PIL Image
        """
        # Load pipeline if not loaded
        self.load_pipeline()

        # Build prompts
        prompt, negative_prompt = self.build_prompt(
            emotion=emotion,
            pose=pose,
            additional_details=additional_details,
            character=character,
        )

        logger.info(f"Generating: {emotion}/{pose}")
        logger.debug(f"Prompt: {prompt}")

        # Setup generator with seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Generate image
        try:
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=self.height,
                width=self.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

            image = result.images[0]
            logger.info("Image generated successfully")

            return image

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def generate_batch(
        self,
        specifications: List[Dict],
        output_dir: str = "data/generated",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        retry_on_failure: bool = True,
        max_retries: int = 3,
    ) -> List[Dict]:
        """
        Generate multiple images from specifications.

        Args:
            specifications: List of dicts with generation parameters
            output_dir: Output directory
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale
            retry_on_failure: Retry with different seeds on failure
            max_retries: Maximum retry attempts

        Returns:
            List of results with paths and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        for i, spec in enumerate(tqdm(specifications, desc="Generating batch")):
            emotion = spec.get("emotion", "neutral")
            pose = spec.get("pose", "front")
            character = spec.get("character", None)
            count = spec.get("count", 1)

            for j in range(count):
                success = False
                attempts = 0

                while not success and attempts <= max_retries:
                    try:
                        # Use specified seed or random
                        seed = spec.get("seed", None)
                        if seed is None or attempts > 0:
                            seed = torch.randint(0, 2**32 - 1, (1,)).item()

                        # Generate image
                        image = self.generate(
                            emotion=emotion,
                            pose=pose,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            character=character,
                        )

                        # Save image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        character_name = character or self.model_info.get("character", "character")
                        filename = f"{character_name}_{emotion}_{pose}_{timestamp}_{j}.png"
                        filepath = output_path / filename

                        image.save(filepath)

                        # Add to results
                        results.append({
                            "filepath": str(filepath),
                            "character": character_name,
                            "emotion": emotion,
                            "pose": pose,
                            "seed": seed,
                            "index": j,
                        })

                        success = True
                        logger.info(f"Saved: {filename}")

                    except Exception as e:
                        attempts += 1
                        logger.error(f"Generation failed (attempt {attempts}): {e}")

                        if not retry_on_failure or attempts > max_retries:
                            logger.error(f"Skipping after {attempts} attempts")
                            break

        logger.info(f"Generated {len(results)} images")
        return results

    def add_to_library(
        self,
        image_path: str,
        character: str,
        emotion: str,
        pose: str = "front",
        quality_score: float = 0.8,
    ):
        """
        Add generated image to the image library database.

        Args:
            image_path: Path to image
            character: Character name
            emotion: Emotion
            pose: Pose
            quality_score: Quality score (0-1)
        """
        try:
            # Import here to avoid circular dependency
            from select_images import ImageSelector

            db_path = os.getenv("IMAGE_LIBRARY_DB_PATH", "data/image_library.db")
            selector = ImageSelector(db_path=db_path)

            selector.add_image(
                character=character.lower(),
                emotion=emotion,
                file_path=image_path,
                source="dreambooth",
                quality_score=quality_score,
                pose=pose,
            )

            logger.info(f"Added to library: {image_path}")

        except Exception as e:
            logger.error(f"Failed to add to library: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate character images using DreamBooth models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate single image
  python scripts/generate_character_image.py --character butcher --emotion happy

  # Generate multiple variations
  python scripts/generate_character_image.py --emotion sarcastic --count 5

  # Generate with specific pose
  python scripts/generate_character_image.py --emotion excited --pose close_up

  # Batch generation from JSON
  python scripts/generate_character_image.py --batch batch_spec.json

  # Generate and add to library
  python scripts/generate_character_image.py --emotion happy --library

  # High quality generation
  python scripts/generate_character_image.py --emotion neutral --steps 50 --guidance 10
        """
    )

    parser.add_argument(
        "--character",
        type=str,
        help="Character name (uses model default if not specified)"
    )

    parser.add_argument(
        "--emotion",
        type=str,
        default="neutral",
        help="Emotion to convey"
    )

    parser.add_argument(
        "--pose",
        type=str,
        default="front",
        help="Pose/angle (front, side, close_up, etc.)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of images to generate (default: 1)"
    )

    parser.add_argument(
        "--batch",
        type=str,
        help="JSON file with batch specifications"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to DreamBooth model (default from .env)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/generated",
        help="Output directory (default: data/generated)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps (default: 30)"
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
        "--library",
        action="store_true",
        help="Add generated images to library database"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )

    parser.add_argument(
        "--details",
        type=str,
        help="Additional prompt details"
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
        generator = CharacterImageGenerator(
            model_path=args.model,
            use_gpu=not args.cpu,
        )
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        sys.exit(1)

    # Batch mode
    if args.batch:
        try:
            with open(args.batch, 'r') as f:
                batch_specs = json.load(f)

            if not isinstance(batch_specs, list):
                logger.error("Batch file must contain a list of specifications")
                sys.exit(1)

            results = generator.generate_batch(
                specifications=batch_specs,
                output_dir=args.output,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )

            # Add to library if requested
            if args.library:
                for result in results:
                    generator.add_to_library(
                        image_path=result["filepath"],
                        character=result["character"],
                        emotion=result["emotion"],
                        pose=result.get("pose", "front"),
                    )

            print(f"\nGenerated {len(results)} images:")
            for result in results:
                print(f"  - {result['filepath']}")

        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            sys.exit(1)

    # Single/multi mode
    else:
        try:
            # Create specifications
            specs = [{
                "character": args.character,
                "emotion": args.emotion,
                "pose": args.pose,
                "count": args.count,
                "seed": args.seed,
            }]

            results = generator.generate_batch(
                specifications=specs,
                output_dir=args.output,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
            )

            # Add to library if requested
            if args.library:
                for result in results:
                    generator.add_to_library(
                        image_path=result["filepath"],
                        character=result["character"],
                        emotion=result["emotion"],
                        pose=result.get("pose", "front"),
                    )

            print(f"\nGenerated {len(results)} images:")
            for result in results:
                print(f"  - {result['filepath']}")

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
