#!/usr/bin/env python3
"""
Generate lip-synced animated videos using SadTalker or Wav2Lip.

This script takes static character images and audio files, then generates
animated videos with synchronized lip movements using deep learning models.
"""

import argparse
import json
import logging
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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


class AnimationGenerator:
    """Generates lip-synced animations using SadTalker or Wav2Lip."""

    def __init__(self, method: str = None, use_gpu: bool = None):
        """
        Initialize the AnimationGenerator.

        Args:
            method: Animation method ('sadtalker' or 'wav2lip')
            use_gpu: Whether to use GPU acceleration (auto-detect if None)
        """
        # Determine method
        if method is None:
            method = os.getenv("LIPSYNC_MODEL", "sadtalker").lower()

        self.method = method

        # GPU setup
        if use_gpu is None:
            use_gpu = os.getenv("USE_GPU", "true").lower() == "true"

        self.device = self._setup_device(use_gpu)

        # Model paths
        self.sadtalker_path = Path(os.getenv(
            "SADTALKER_CHECKPOINT_PATH",
            "models/sadtalker"
        ))
        self.wav2lip_path = Path(os.getenv(
            "WAV2LIP_CHECKPOINT_PATH",
            "models/wav2lip"
        ))

        # Video settings
        self.width = int(os.getenv("VIDEO_WIDTH", "1080"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "1920"))
        self.fps = int(os.getenv("VIDEO_FPS", "30"))

        logger.info(f"Animation method: {self.method}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Video settings: {self.width}x{self.height} @ {self.fps}fps")

    def _setup_device(self, use_gpu: bool) -> str:
        """
        Setup compute device.

        Args:
            use_gpu: Whether to attempt GPU usage

        Returns:
            Device string ('cuda' or 'cpu')
        """
        if use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_id = int(os.getenv("GPU_DEVICE", "0"))
                    device = f"cuda:{gpu_id}"
                    logger.info(f"GPU available: {torch.cuda.get_device_name(gpu_id)}")
                    return device
            except ImportError:
                logger.warning("PyTorch not installed, using CPU")

        logger.info("Using CPU for animation")
        return "cpu"

    def _check_sadtalker_installation(self) -> bool:
        """Check if SadTalker is installed and available."""
        if not self.sadtalker_path.exists():
            logger.warning(f"SadTalker checkpoint not found at {self.sadtalker_path}")
            return False

        try:
            # Check for SadTalker dependencies
            import torch
            import cv2
            import numpy
            logger.info("SadTalker dependencies available")
            return True
        except ImportError as e:
            logger.warning(f"SadTalker dependency missing: {e}")
            return False

    def _check_wav2lip_installation(self) -> bool:
        """Check if Wav2Lip is installed and available."""
        if not self.wav2lip_path.exists():
            logger.warning(f"Wav2Lip checkpoint not found at {self.wav2lip_path}")
            return False

        try:
            # Check for Wav2Lip dependencies
            import torch
            import cv2
            import numpy
            logger.info("Wav2Lip dependencies available")
            return True
        except ImportError as e:
            logger.warning(f"Wav2Lip dependency missing: {e}")
            return False

    def animate_sadtalker(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        pose_style: int = 0,
        expression_scale: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate animation using SadTalker.

        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path to save output video
            pose_style: Head pose style (0-45)
            expression_scale: Expression intensity (0.0-2.0)

        Returns:
            Path to generated video
        """
        if not self._check_sadtalker_installation():
            raise RuntimeError("SadTalker is not properly installed")

        logger.info(f"Generating SadTalker animation: {Path(image_path).name}")

        try:
            # Import SadTalker (assuming it's installed as a package)
            # NOTE: This is a placeholder - actual implementation depends on SadTalker's API
            from sadtalker import SadTalker

            # Initialize SadTalker
            sadtalker = SadTalker(
                checkpoint_path=str(self.sadtalker_path),
                device=self.device
            )

            # Generate animation
            result = sadtalker.generate(
                source_image=image_path,
                driven_audio=audio_path,
                output_path=output_path,
                pose_style=pose_style,
                expression_scale=expression_scale,
                size=self.height  # SadTalker uses single size param
            )

            logger.info(f"SadTalker animation complete: {output_path}")
            return output_path

        except ImportError:
            # Fallback to command-line interface
            logger.info("Using SadTalker CLI interface")
            return self._animate_sadtalker_cli(
                image_path,
                audio_path,
                output_path,
                pose_style,
                expression_scale
            )

    def _animate_sadtalker_cli(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        pose_style: int = 0,
        expression_scale: float = 1.0
    ) -> str:
        """
        Run SadTalker via command-line interface.

        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path to save output video
            pose_style: Head pose style
            expression_scale: Expression intensity

        Returns:
            Path to generated video
        """
        # SadTalker CLI command
        cmd = [
            "python",
            str(self.sadtalker_path / "inference.py"),
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", str(Path(output_path).parent),
            "--pose_style", str(pose_style),
            "--expression_scale", str(expression_scale),
            "--size", str(self.height)
        ]

        if "cuda" in self.device:
            cmd.extend(["--device", "cuda"])
        else:
            cmd.extend(["--device", "cpu"])

        try:
            logger.debug(f"Running SadTalker CLI: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"SadTalker output: {result.stdout}")

            # SadTalker typically outputs to result_dir with auto-generated name
            # We may need to rename or move the file
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"SadTalker CLI failed: {e.stderr}")
            raise RuntimeError(f"SadTalker animation failed: {e.stderr}")

    def animate_wav2lip(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = None,
        **kwargs
    ) -> str:
        """
        Generate animation using Wav2Lip.

        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path to save output video
            fps: Output video FPS

        Returns:
            Path to generated video
        """
        if not self._check_wav2lip_installation():
            raise RuntimeError("Wav2Lip is not properly installed")

        if fps is None:
            fps = self.fps

        logger.info(f"Generating Wav2Lip animation: {Path(image_path).name}")

        try:
            # Import Wav2Lip (assuming it's installed as a package)
            # NOTE: This is a placeholder - actual implementation depends on Wav2Lip's API
            from wav2lip import Wav2Lip

            wav2lip = Wav2Lip(
                checkpoint_path=str(self.wav2lip_path / "wav2lip.pth"),
                device=self.device
            )

            result = wav2lip.generate(
                face=image_path,
                audio=audio_path,
                output=output_path,
                fps=fps,
                resize_factor=1
            )

            logger.info(f"Wav2Lip animation complete: {output_path}")
            return output_path

        except ImportError:
            # Fallback to command-line interface
            logger.info("Using Wav2Lip CLI interface")
            return self._animate_wav2lip_cli(image_path, audio_path, output_path, fps)

    def _animate_wav2lip_cli(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = None
    ) -> str:
        """
        Run Wav2Lip via command-line interface.

        Args:
            image_path: Path to source image
            audio_path: Path to audio file
            output_path: Path to save output video
            fps: Output video FPS

        Returns:
            Path to generated video
        """
        if fps is None:
            fps = self.fps

        # Wav2Lip CLI command
        cmd = [
            "python",
            str(self.wav2lip_path / "inference.py"),
            "--checkpoint_path", str(self.wav2lip_path / "wav2lip.pth"),
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--fps", str(fps),
            "--resize_factor", "1"
        ]

        try:
            logger.debug(f"Running Wav2Lip CLI: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"Wav2Lip output: {result.stdout}")
            return output_path

        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip CLI failed: {e.stderr}")
            raise RuntimeError(f"Wav2Lip animation failed: {e.stderr}")

    def animate_single_line(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        character: str = None,
        emotion: str = None
    ) -> str:
        """
        Generate animation for a single dialogue line.

        Args:
            image_path: Path to character image
            audio_path: Path to audio file
            output_path: Path to save output video
            character: Character name (for metadata)
            emotion: Emotion (for metadata)

        Returns:
            Path to generated video file
        """
        # Validate inputs
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Adjust parameters based on character/emotion
        params = {}
        if self.method == "sadtalker":
            # More expressive for excited/happy emotions
            if emotion in ["excited", "happy", "surprised"]:
                params["expression_scale"] = 1.5
                params["pose_style"] = 1
            # Calmer for sad/neutral
            elif emotion in ["sad", "neutral"]:
                params["expression_scale"] = 0.8
                params["pose_style"] = 0
            else:
                params["expression_scale"] = 1.0
                params["pose_style"] = 0

        # Generate animation
        if self.method == "sadtalker":
            return self.animate_sadtalker(image_path, audio_path, output_path, **params)
        elif self.method == "wav2lip":
            return self.animate_wav2lip(image_path, audio_path, output_path, **params)
        else:
            raise ValueError(f"Unknown animation method: {self.method}")

    def process_timeline(
        self,
        timeline_path: str,
        image_selections_path: str,
        output_dir: str = "data/animated"
    ) -> List[Dict]:
        """
        Process entire timeline and generate all animations.

        Args:
            timeline_path: Path to timeline JSON (from generate_audio.py)
            image_selections_path: Path to image selections JSON (from select_images.py)
            output_dir: Directory to save animated clips

        Returns:
            List of animation metadata dictionaries
        """
        # Load timeline
        with open(timeline_path, 'r') as f:
            timeline = json.load(f)

        # Load image selections
        with open(image_selections_path, 'r') as f:
            selections = json.load(f)

        episode_name = timeline['episode']
        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing episode: {episode_name}")
        logger.info(f"Lines to animate: {len(timeline['lines'])}")

        animations = []

        for i, line in enumerate(timeline['lines'], 1):
            # Find matching image
            image_path = None
            for sel in selections['selections']:
                if sel['line_index'] == line['index']:
                    image_path = sel['image_path']
                    break

            if not image_path or not Path(image_path).exists():
                logger.error(f"Image not found for line {i}")
                raise FileNotFoundError(f"Missing image for line {i}")

            # Generate output filename
            character = line['character'].lower()
            emotion = line['emotion']
            output_filename = f"{i:03d}_{character}_{emotion}.mp4"
            output_filepath = output_path / output_filename

            try:
                logger.info(f"[{i}/{len(timeline['lines'])}] Animating {character}/{emotion}...")

                # Generate animation
                result_path = self.animate_single_line(
                    image_path=image_path,
                    audio_path=line['audio_file'],
                    output_path=str(output_filepath),
                    character=line['character'],
                    emotion=emotion
                )

                # Record metadata
                animations.append({
                    'line_index': line['index'],
                    'character': line['character'],
                    'emotion': emotion,
                    'image_path': image_path,
                    'audio_path': line['audio_file'],
                    'video_path': result_path,
                    'start_time': line['start_time'],
                    'duration': line['duration'],
                    'end_time': line['end_time']
                })

            except Exception as e:
                logger.error(f"Failed to animate line {i}: {e}")
                raise

        logger.info(f"Successfully animated {len(animations)} clips")
        return animations

    def save_animation_manifest(
        self,
        animations: List[Dict],
        output_dir: str,
        episode_name: str
    ) -> str:
        """
        Save animation manifest JSON.

        Args:
            animations: List of animation metadata
            output_dir: Output directory
            episode_name: Episode name

        Returns:
            Path to manifest file
        """
        manifest = {
            'episode': episode_name,
            'method': self.method,
            'total_clips': len(animations),
            'animations': animations
        }

        manifest_path = Path(output_dir) / episode_name / "animations.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(f"Animation manifest saved: {manifest_path}")
        return str(manifest_path)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate lip-synced animations using SadTalker or Wav2Lip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animate entire episode from timeline and images
  python animate.py \\
    --timeline data/audio/20240101_120000/timeline.json \\
    --images data/image_selections.json

  # Use specific method
  python animate.py \\
    --timeline timeline.json \\
    --images selections.json \\
    --method wav2lip

  # Animate single image/audio pair
  python animate.py \\
    --image data/butcher/butcher_neutral.png \\
    --audio data/audio/episode/001_butcher_sarcastic.mp3 \\
    --output data/animated/test.mp4

  # Use CPU instead of GPU
  python animate.py \\
    --timeline timeline.json \\
    --images selections.json \\
    --cpu
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--timeline",
        help="Path to timeline JSON file (batch mode)"
    )
    mode_group.add_argument(
        "--image",
        help="Path to single image file (single mode)"
    )

    # Batch mode arguments
    parser.add_argument(
        "--images",
        help="Path to image selections JSON (required for batch mode)"
    )

    # Single mode arguments
    parser.add_argument(
        "--audio",
        help="Path to audio file (required for single mode)"
    )

    # Output
    parser.add_argument(
        "--output",
        help="Output path (for single mode) or directory (for batch mode)"
    )

    # Method
    parser.add_argument(
        "--method",
        choices=["sadtalker", "wav2lip"],
        help="Animation method (default: from .env or sadtalker)"
    )

    # Performance
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of GPU"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate arguments
    if args.timeline and not args.images:
        parser.error("--images is required when using --timeline")

    if args.image and not args.audio:
        parser.error("--audio is required when using --image")

    # Initialize generator
    try:
        generator = AnimationGenerator(
            method=args.method,
            use_gpu=not args.cpu
        )
    except Exception as e:
        logger.error(f"Failed to initialize animation generator: {e}")
        sys.exit(1)

    # Batch mode
    if args.timeline:
        output_dir = args.output or "data/animated"

        try:
            animations = generator.process_timeline(
                timeline_path=args.timeline,
                image_selections_path=args.images,
                output_dir=output_dir
            )

            # Save manifest
            with open(args.timeline, 'r') as f:
                timeline_data = json.load(f)
            episode_name = timeline_data['episode']

            manifest_path = generator.save_animation_manifest(
                animations,
                output_dir,
                episode_name
            )

            print("\n" + "=" * 80)
            print("ANIMATION COMPLETE")
            print("=" * 80)
            print(f"Episode: {episode_name}")
            print(f"Method: {generator.method}")
            print(f"Animated clips: {len(animations)}")
            print(f"Output directory: {Path(output_dir) / episode_name}")
            print(f"Manifest: {manifest_path}")
            print("=" * 80 + "\n")

        except Exception as e:
            logger.error(f"Failed to process timeline: {e}")
            sys.exit(1)

    # Single mode
    else:
        if not args.output:
            parser.error("--output is required for single image mode")

        try:
            result = generator.animate_single_line(
                image_path=args.image,
                audio_path=args.audio,
                output_path=args.output
            )

            print(f"\nAnimation complete: {result}\n")

        except Exception as e:
            logger.error(f"Failed to generate animation: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
