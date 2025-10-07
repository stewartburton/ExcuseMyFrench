#!/usr/bin/env python3
"""
Training Data Preparation Script

This script prepares and organizes training images for DreamBooth training.

Features:
- Image validation and quality checks
- Automatic resizing and cropping
- Caption generation
- Dataset organization
- Data augmentation (optional)
- Duplicate detection

Usage:
    python scripts/prepare_training_data.py --input photos/butcher --output training/butcher/images
    python scripts/prepare_training_data.py --character butcher --validate
"""

import argparse
import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image, ImageOps, ImageFilter
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


class ImageValidator:
    """Validates and checks quality of training images."""

    def __init__(self, min_resolution: int = 512, max_aspect_ratio: float = 2.0):
        """
        Initialize validator.

        Args:
            min_resolution: Minimum width/height in pixels
            max_aspect_ratio: Maximum aspect ratio (width/height or height/width)
        """
        self.min_resolution = min_resolution
        self.max_aspect_ratio = max_aspect_ratio

    def validate(self, image_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate an image for training.

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check if file exists
            if not image_path.exists():
                return False, "File does not exist"

            # Try to open image
            try:
                image = Image.open(image_path)
            except Exception as e:
                return False, f"Cannot open image: {e}"

            # Check format
            if image.format not in ['JPEG', 'PNG', 'WEBP']:
                return False, f"Unsupported format: {image.format}"

            # Check mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                return False, f"Unsupported mode: {image.mode}"

            # Check resolution
            width, height = image.size
            if width < self.min_resolution or height < self.min_resolution:
                return False, f"Resolution too low: {width}x{height} (minimum {self.min_resolution})"

            # Check aspect ratio
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > self.max_aspect_ratio:
                return False, f"Aspect ratio too extreme: {aspect_ratio:.2f} (maximum {self.max_aspect_ratio})"

            # Check if image is corrupted
            try:
                image.verify()
            except Exception as e:
                return False, f"Image appears corrupted: {e}"

            return True, None

        except Exception as e:
            return False, f"Validation error: {e}"


class ImageProcessor:
    """Processes images for training."""

    def __init__(
        self,
        target_resolution: int = 512,
        center_crop: bool = True,
        auto_orient: bool = True,
    ):
        """
        Initialize processor.

        Args:
            target_resolution: Target resolution for training images
            center_crop: Whether to center crop to square
            auto_orient: Auto-orient based on EXIF data
        """
        self.target_resolution = target_resolution
        self.center_crop = center_crop
        self.auto_orient = auto_orient

    def process(self, image_path: Path, output_path: Path) -> bool:
        """
        Process an image and save to output path.

        Args:
            image_path: Input image path
            output_path: Output image path

        Returns:
            True if successful, False otherwise
        """
        try:
            # Open image
            image = Image.open(image_path)

            # Auto-orient based on EXIF
            if self.auto_orient:
                image = ImageOps.exif_transpose(image)

            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Center crop to square if requested
            if self.center_crop:
                width, height = image.size
                crop_size = min(width, height)
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                image = image.crop((left, top, right, bottom))

            # Resize to target resolution
            image = image.resize(
                (self.target_resolution, self.target_resolution),
                Image.LANCZOS
            )

            # Save processed image
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, 'PNG', quality=95)

            return True

        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return False


class DataAugmenter:
    """Augments training data with variations."""

    def __init__(self, target_resolution: int = 512):
        """
        Initialize augmenter.

        Args:
            target_resolution: Resolution for augmented images
        """
        self.target_resolution = target_resolution

    def augment(self, image_path: Path, output_dir: Path, num_variations: int = 3) -> List[Path]:
        """
        Create augmented variations of an image.

        Args:
            image_path: Input image path
            output_dir: Output directory for augmented images
            num_variations: Number of variations to create

        Returns:
            List of paths to augmented images
        """
        augmented_paths = []

        try:
            image = Image.open(image_path)

            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Create variations
            variations = [
                ("original", image),
                ("flip", image.transpose(Image.FLIP_LEFT_RIGHT)),
                ("slight_rotation", image.rotate(5, fillcolor=(255, 255, 255))),
                ("adjust_brightness", self._adjust_brightness(image, 1.1)),
                ("adjust_contrast", self._adjust_contrast(image, 1.1)),
            ]

            # Save up to num_variations
            for i, (name, var_image) in enumerate(variations[:num_variations]):
                output_path = output_dir / f"{image_path.stem}_{name}.png"
                var_image.resize(
                    (self.target_resolution, self.target_resolution),
                    Image.LANCZOS
                ).save(output_path, 'PNG')
                augmented_paths.append(output_path)

            return augmented_paths

        except Exception as e:
            logger.error(f"Failed to augment {image_path}: {e}")
            return []

    def _adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust image brightness."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def _adjust_contrast(self, image: Image.Image, factor: float) -> Image.Image:
        """Adjust image contrast."""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)


class CaptionGenerator:
    """Generates captions for training images."""

    def __init__(self, instance_token: str = "sks", class_name: str = "dog"):
        """
        Initialize caption generator.

        Args:
            instance_token: Unique identifier token (e.g., "sks")
            class_name: Class name (e.g., "dog", "cat")
        """
        self.instance_token = instance_token
        self.class_name = class_name

    def generate(
        self,
        image_path: Path,
        character_name: Optional[str] = None,
        emotion: Optional[str] = None,
    ) -> str:
        """
        Generate caption for an image.

        Args:
            image_path: Path to image
            character_name: Character name
            emotion: Emotion if known

        Returns:
            Generated caption
        """
        # Base caption with instance token
        caption = f"a photo of {self.instance_token} {self.class_name}"

        # Add character details if available
        if character_name:
            caption = f"a photo of {self.instance_token} {character_name}"

        # Add emotion if specified
        if emotion:
            caption += f", {emotion} expression"

        return caption


class DuplicateDetector:
    """Detects duplicate or very similar images."""

    def __init__(self):
        """Initialize duplicate detector."""
        self.image_hashes: Dict[str, Path] = {}

    def compute_hash(self, image_path: Path) -> str:
        """
        Compute perceptual hash of an image.

        Args:
            image_path: Path to image

        Returns:
            Hash string
        """
        try:
            # Open and resize to small size for comparison
            image = Image.open(image_path)
            image = image.convert('L')  # Grayscale
            image = image.resize((8, 8), Image.LANCZOS)

            # Get pixel data
            pixels = list(image.getdata())

            # Compute average
            avg = sum(pixels) / len(pixels)

            # Create hash based on pixels above/below average
            hash_str = ''.join('1' if p > avg else '0' for p in pixels)

            return hash_str

        except Exception as e:
            logger.error(f"Failed to compute hash for {image_path}: {e}")
            return ""

    def is_duplicate(self, image_path: Path, threshold: int = 5) -> Tuple[bool, Optional[Path]]:
        """
        Check if image is a duplicate.

        Args:
            image_path: Path to check
            threshold: Hamming distance threshold for duplicates

        Returns:
            Tuple of (is_duplicate, original_path)
        """
        current_hash = self.compute_hash(image_path)

        if not current_hash:
            return False, None

        # Check against existing hashes
        for existing_hash, existing_path in self.image_hashes.items():
            # Compute hamming distance
            distance = sum(c1 != c2 for c1, c2 in zip(current_hash, existing_hash))

            if distance <= threshold:
                return True, existing_path

        # Not a duplicate, add to known hashes
        self.image_hashes[current_hash] = image_path

        return False, None


class TrainingDataPreparer:
    """Main class for preparing training data."""

    def __init__(
        self,
        character: str,
        instance_token: str = "sks",
        class_name: str = "dog",
        resolution: int = 512,
    ):
        """
        Initialize data preparer.

        Args:
            character: Character name
            instance_token: Unique identifier token
            class_name: Class name
            resolution: Target resolution
        """
        self.character = character
        self.instance_token = instance_token
        self.class_name = class_name
        self.resolution = resolution

        # Initialize components
        self.validator = ImageValidator(min_resolution=resolution // 2)
        self.processor = ImageProcessor(target_resolution=resolution)
        self.augmenter = DataAugmenter(target_resolution=resolution)
        self.caption_gen = CaptionGenerator(instance_token, class_name)
        self.duplicate_detector = DuplicateDetector()

        logger.info(f"Data preparer initialized for {character}")

    def prepare_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augment: bool = False,
        num_augmentations: int = 3,
        check_duplicates: bool = True,
    ) -> Dict[str, any]:
        """
        Prepare training dataset from input images.

        Args:
            input_dir: Directory containing source images
            output_dir: Output directory for processed images
            augment: Whether to create augmented variations
            num_augmentations: Number of augmentations per image
            check_duplicates: Whether to check for duplicates

        Returns:
            Dictionary with preparation statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        logger.info(f"Preparing dataset from {input_path}")
        logger.info(f"Output directory: {output_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_files)} images")

        # Statistics
        stats = {
            'total_input': len(image_files),
            'valid': 0,
            'invalid': 0,
            'duplicates': 0,
            'processed': 0,
            'augmented': 0,
        }

        # Process each image
        valid_images = []
        for image_path in tqdm(image_files, desc="Validating images"):
            # Validate
            is_valid, error = self.validator.validate(image_path)

            if not is_valid:
                logger.warning(f"Invalid image {image_path.name}: {error}")
                stats['invalid'] += 1
                continue

            # Check for duplicates
            if check_duplicates:
                is_dup, original = self.duplicate_detector.is_duplicate(image_path)
                if is_dup:
                    logger.info(f"Duplicate image: {image_path.name} (similar to {original.name})")
                    stats['duplicates'] += 1
                    continue

            stats['valid'] += 1
            valid_images.append(image_path)

        logger.info(f"Valid images: {stats['valid']}")
        logger.info(f"Invalid images: {stats['invalid']}")
        logger.info(f"Duplicates: {stats['duplicates']}")

        # Process valid images
        captions = {}
        for i, image_path in enumerate(tqdm(valid_images, desc="Processing images")):
            # Generate output filename
            output_filename = f"{self.character}_{i:04d}.png"
            output_image_path = output_path / output_filename

            # Process image
            if self.processor.process(image_path, output_image_path):
                stats['processed'] += 1

                # Generate caption
                caption = self.caption_gen.generate(image_path, self.character)
                captions[output_filename] = caption

                # Augment if requested
                if augment:
                    aug_paths = self.augmenter.augment(
                        output_image_path,
                        output_path,
                        num_augmentations
                    )
                    stats['augmented'] += len(aug_paths)

                    # Add captions for augmented images
                    for aug_path in aug_paths:
                        captions[aug_path.name] = caption

        # Save captions
        captions_file = output_path / "captions.txt"
        with open(captions_file, 'w') as f:
            for filename, caption in sorted(captions.items()):
                f.write(f"{filename}\t{caption}\n")

        logger.info(f"Processed {stats['processed']} images")
        logger.info(f"Created {stats['augmented']} augmented images")
        logger.info(f"Total training images: {stats['processed'] + stats['augmented']}")
        logger.info(f"Captions saved to {captions_file}")

        return stats

    def validate_dataset(self, dataset_dir: str) -> Dict[str, any]:
        """
        Validate an existing dataset.

        Args:
            dataset_dir: Directory containing dataset

        Returns:
            Dictionary with validation results
        """
        dataset_path = Path(dataset_dir)

        logger.info(f"Validating dataset: {dataset_path}")

        # Check if directory exists
        if not dataset_path.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            return {'valid': False, 'error': 'Directory not found'}

        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        images = []
        for ext in image_extensions:
            images.extend(dataset_path.glob(f"*{ext}"))
            images.extend(dataset_path.glob(f"*{ext.upper()}"))

        # Validate each image
        stats = {
            'total_images': len(images),
            'valid_images': 0,
            'invalid_images': 0,
            'min_resolution': float('inf'),
            'max_resolution': 0,
            'avg_resolution': 0,
            'has_captions': False,
        }

        total_pixels = 0
        for image_path in tqdm(images, desc="Validating"):
            is_valid, error = self.validator.validate(image_path)

            if is_valid:
                stats['valid_images'] += 1

                # Get resolution
                with Image.open(image_path) as img:
                    width, height = img.size
                    resolution = min(width, height)
                    stats['min_resolution'] = min(stats['min_resolution'], resolution)
                    stats['max_resolution'] = max(stats['max_resolution'], resolution)
                    total_pixels += width * height
            else:
                stats['invalid_images'] += 1
                logger.warning(f"Invalid: {image_path.name} - {error}")

        if stats['valid_images'] > 0:
            stats['avg_resolution'] = int((total_pixels / stats['valid_images']) ** 0.5)

        # Check for captions file
        captions_file = dataset_path / "captions.txt"
        stats['has_captions'] = captions_file.exists()

        # Print summary
        logger.info("\nDataset Validation Summary:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Valid images: {stats['valid_images']}")
        logger.info(f"  Invalid images: {stats['invalid_images']}")
        logger.info(f"  Resolution range: {stats['min_resolution']} - {stats['max_resolution']}")
        logger.info(f"  Average resolution: {stats['avg_resolution']}")
        logger.info(f"  Has captions: {'Yes' if stats['has_captions'] else 'No'}")

        # Recommendations
        logger.info("\nRecommendations:")
        if stats['valid_images'] < 5:
            logger.warning("  ⚠ Very few images (< 5). Add more for better training.")
        elif stats['valid_images'] < 10:
            logger.info("  ℹ Consider adding more images for better results (10-20 recommended).")
        else:
            logger.info("  ✓ Good number of training images.")

        if stats['min_resolution'] < self.resolution:
            logger.warning(f"  ⚠ Some images below target resolution ({self.resolution}px)")

        if not stats['has_captions']:
            logger.warning("  ⚠ No captions.txt file found")

        return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare training data for DreamBooth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training data
  python scripts/prepare_training_data.py --input photos/butcher --output training/butcher/images --character butcher

  # Prepare with augmentation
  python scripts/prepare_training_data.py --input photos/butcher --output training/butcher/images --augment

  # Validate existing dataset
  python scripts/prepare_training_data.py --validate --dataset training/butcher/images

  # Prepare with custom settings
  python scripts/prepare_training_data.py --input photos/butcher --output training/butcher/images --resolution 768 --class-name "french bulldog"
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Input directory containing source images"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for processed images"
    )

    parser.add_argument(
        "--character",
        type=str,
        default="butcher",
        help="Character name (default: butcher)"
    )

    parser.add_argument(
        "--instance-token",
        type=str,
        default="sks",
        help="Unique instance token (default: sks)"
    )

    parser.add_argument(
        "--class-name",
        type=str,
        default="dog",
        help="Class name (default: dog)"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target resolution (default: 512)"
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Create augmented variations"
    )

    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=3,
        help="Number of augmentations per image (default: 3)"
    )

    parser.add_argument(
        "--no-duplicate-check",
        action="store_true",
        help="Skip duplicate detection"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing dataset"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset directory to validate"
    )

    args = parser.parse_args()

    # Create preparer
    preparer = TrainingDataPreparer(
        character=args.character,
        instance_token=args.instance_token,
        class_name=args.class_name,
        resolution=args.resolution,
    )

    if args.validate:
        # Validate mode
        dataset_dir = args.dataset or args.output or f"training/{args.character}/images"
        stats = preparer.validate_dataset(dataset_dir)
    elif args.input and args.output:
        # Prepare mode
        stats = preparer.prepare_dataset(
            input_dir=args.input,
            output_dir=args.output,
            augment=args.augment,
            num_augmentations=args.num_augmentations,
            check_duplicates=not args.no_duplicate_check,
        )

        print("\n" + "="*60)
        print("Dataset Preparation Complete!")
        print("="*60)
        print(f"Total input images: {stats['total_input']}")
        print(f"Valid images: {stats['valid']}")
        print(f"Invalid images: {stats['invalid']}")
        print(f"Duplicates removed: {stats['duplicates']}")
        print(f"Processed images: {stats['processed']}")
        print(f"Augmented images: {stats['augmented']}")
        print(f"Total training images: {stats['processed'] + stats['augmented']}")
        print("\nNext steps:")
        print(f"  1. Review images in: {args.output}")
        print(f"  2. Edit captions if needed: {args.output}/captions.txt")
        print(f"  3. Run training: python scripts/train_dreambooth.py --character {args.character}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
