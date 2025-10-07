#!/usr/bin/env python3
"""
Add lip-sync animation to static character images.

This script generates lip-synced videos from static images and audio files using
either SadTalker or Wav2Lip animation methods. It can process individual image-audio
pairs or entire timelines from the generate_audio.py pipeline.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Generates lip-synced animations from static images and audio."""

    def __init__(self, method: str = "sadtalker"):
        """
        Initialize the AnimationGenerator.

        Args:
            method: Animation method to use ("sadtalker" or "wav2lip")

        Raises:
            ValueError: If method is not supported
        """
        if method not in ["sadtalker", "wav2lip"]:
            raise ValueError(f"Unsupported animation method: {method}. Use 'sadtalker' or 'wav2lip'")

        self.method = method
        self.output_width = int(os.getenv("VIDEO_WIDTH", "1080"))
        self.output_height = int(os.getenv("VIDEO_HEIGHT", "1920"))
        self.fps = int(os.getenv("VIDEO_FPS", "30"))

        # Method-specific settings
        if self.method == "sadtalker":
            self.sadtalker_checkpoint = os.getenv("SADTALKER_CHECKPOINT_PATH", "models/sadtalker")
            self.sadtalker_config = os.getenv("SADTALKER_CONFIG_PATH", "models/sadtalker/config")
            self.sadtalker_enhancer = os.getenv("SADTALKER_ENHANCER", "gfpgan")
            self.sadtalker_preprocess = os.getenv("SADTALKER_PREPROCESS", "crop")
            logger.info(f"SadTalker checkpoint: {self.sadtalker_checkpoint}")
        elif self.method == "wav2lip":
            self.wav2lip_checkpoint = os.getenv("WAV2LIP_CHECKPOINT_PATH", "models/wav2lip/wav2lip.pth")
            self.wav2lip_face_detection = os.getenv("WAV2LIP_FACE_DETECTION", "s3fd")
            self.wav2lip_resize_factor = int(os.getenv("WAV2LIP_RESIZE_FACTOR", "1"))
            logger.info(f"Wav2Lip checkpoint: {self.wav2lip_checkpoint}")

        # Quality settings
        self.quality = os.getenv("ANIMATION_QUALITY", "high")  # low, medium, high
        self.batch_size = int(os.getenv("ANIMATION_BATCH_SIZE", "8"))

        logger.info(f"Animation method: {self.method}")
        logger.info(f"Output resolution: {self.output_width}x{self.output_height} @ {self.fps}fps")

    def _validate_file_path(self, file_path: str, base_dir: str = "data") -> bool:
        """
        Validate that a file path is within the allowed directory.

        Args:
            file_path: Path to validate
            base_dir: Base directory to restrict to

        Returns:
            True if path is valid, False otherwise
        """
        try:
            # Get the base directory (resolve to absolute path)
            base = Path(base_dir).resolve()

            # Resolve the file path to its canonical absolute path
            resolved_path = Path(file_path).resolve()

            # Check if the resolved path is within the base directory
            if hasattr(resolved_path, 'is_relative_to'):
                # Python 3.9+
                return resolved_path.is_relative_to(base)
            else:
                # Python 3.8 fallback
                try:
                    resolved_path.relative_to(base)
                    return True
                except ValueError:
                    return False
        except Exception as e:
            logger.error(f"Error validating path {file_path}: {e}")
            return False

    def _check_dependencies(self) -> bool:
        """
        Check if required dependencies are available.

        Returns:
            True if all dependencies are available, False otherwise
        """
        # Check ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("FFmpeg not found. Please install FFmpeg and add it to PATH")
            return False

        # Check method-specific dependencies
        if self.method == "sadtalker":
            if not Path(self.sadtalker_checkpoint).exists():
                logger.error(f"SadTalker checkpoint not found: {self.sadtalker_checkpoint}")
                logger.info("Download SadTalker models from: https://github.com/OpenTalker/SadTalker")
                return False
        elif self.method == "wav2lip":
            if not Path(self.wav2lip_checkpoint).exists():
                logger.error(f"Wav2Lip checkpoint not found: {self.wav2lip_checkpoint}")
                logger.info("Download Wav2Lip models from: https://github.com/Rudrabha/Wav2Lip")
                return False

        return True

    def animate_image(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        character: Optional[str] = None
    ) -> str:
        """
        Animate a single image with audio using lip-sync.

        Args:
            image_path: Path to the static image file
            audio_path: Path to the audio file
            output_path: Path to save the animated video
            character: Character name (for logging/metadata)

        Returns:
            Path to the generated animated video

        Raises:
            FileNotFoundError: If image or audio file not found
            ValueError: If paths are invalid
        """
        # Validate input paths
        if not self._validate_file_path(image_path):
            raise ValueError(f"Invalid image path (not within data/ directory): {image_path}")

        if not self._validate_file_path(audio_path):
            raise ValueError(f"Invalid audio path (not within data/ directory): {audio_path}")

        image_file = Path(image_path)
        audio_file = Path(audio_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Create output directory
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Log operation
        char_info = f" for {character}" if character else ""
        logger.info(f"Animating image{char_info}: {image_file.name}")
        logger.debug(f"Image: {image_path}")
        logger.debug(f"Audio: {audio_path}")
        logger.debug(f"Output: {output_path}")

        # Call appropriate animation method
        if self.method == "sadtalker":
            self._animate_sadtalker(image_path, audio_path, output_path)
        elif self.method == "wav2lip":
            self._animate_wav2lip(image_path, audio_path, output_path)

        # Verify output was created
        if not output_file.exists():
            raise RuntimeError(f"Animation failed to create output file: {output_path}")

        # Resize/reformat if necessary
        self._post_process_video(output_path, output_path)

        logger.info(f"Animation complete: {output_file.name}")
        return str(output_file)

    def _animate_sadtalker(
        self,
        image_path: str,
        audio_path: str,
        output_path: str
    ):
        """
        Animate using SadTalker.

        Args:
            image_path: Path to the static image file
            audio_path: Path to the audio file
            output_path: Path to save the animated video
        """
        try:
            # Import SadTalker (assumed to be installed as a package or in PYTHONPATH)
            try:
                from sadtalker.inference import SadTalker
            except ImportError:
                logger.error("SadTalker not installed or not in PYTHONPATH")
                logger.info("Clone SadTalker and add to PYTHONPATH: https://github.com/OpenTalker/SadTalker")
                raise

            # Initialize SadTalker
            sadtalker = SadTalker(
                checkpoint_path=self.sadtalker_checkpoint,
                config_path=self.sadtalker_config,
                lazy_load=True
            )

            # Set generation parameters based on quality
            if self.quality == "high":
                size = 512
                pose_style = 0  # Natural pose
            elif self.quality == "medium":
                size = 256
                pose_style = 0
            else:  # low
                size = 256
                pose_style = 1  # More stable pose

            # Generate animation
            result = sadtalker.test(
                source_image=image_path,
                driven_audio=audio_path,
                preprocess=self.sadtalker_preprocess,
                still_mode=True,
                use_enhancer=self.sadtalker_enhancer if self.quality == "high" else None,
                batch_size=self.batch_size,
                size=size,
                pose_style=pose_style,
                result_dir=str(Path(output_path).parent)
            )

            # SadTalker saves with its own naming convention, move to expected location
            if result and Path(result).exists():
                import shutil
                shutil.move(result, output_path)

        except ImportError:
            # Fallback to command-line interface if package not available
            logger.warning("Using SadTalker CLI instead of Python API")
            self._animate_sadtalker_cli(image_path, audio_path, output_path)

    def _animate_sadtalker_cli(
        self,
        image_path: str,
        audio_path: str,
        output_path: str
    ):
        """
        Animate using SadTalker command-line interface.

        Args:
            image_path: Path to the static image file
            audio_path: Path to the audio file
            output_path: Path to save the animated video
        """
        # Construct SadTalker command
        sadtalker_script = os.getenv("SADTALKER_SCRIPT_PATH", "sadtalker/inference.py")

        cmd = [
            sys.executable,
            sadtalker_script,
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", str(Path(output_path).parent),
            "--checkpoint_dir", self.sadtalker_checkpoint,
            "--still",
            "--preprocess", self.sadtalker_preprocess,
            "--batch_size", str(self.batch_size)
        ]

        if self.quality == "high" and self.sadtalker_enhancer:
            cmd.extend(["--enhancer", self.sadtalker_enhancer])

        logger.debug(f"Running SadTalker: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"SadTalker output: {result.stdout}")

            # Find generated video and move to expected location
            result_dir = Path(output_path).parent
            generated_videos = list(result_dir.glob("*.mp4"))

            if generated_videos:
                import shutil
                shutil.move(str(generated_videos[0]), output_path)
            else:
                raise RuntimeError("SadTalker did not generate output video")

        except subprocess.CalledProcessError as e:
            logger.error(f"SadTalker CLI error: {e.stderr}")
            raise RuntimeError(f"SadTalker animation failed: {e.stderr}")

    def _animate_wav2lip(
        self,
        image_path: str,
        audio_path: str,
        output_path: str
    ):
        """
        Animate using Wav2Lip.

        Args:
            image_path: Path to the static image file
            audio_path: Path to the audio file
            output_path: Path to save the animated video
        """
        # Wav2Lip typically requires command-line interface
        wav2lip_script = os.getenv("WAV2LIP_SCRIPT_PATH", "wav2lip/inference.py")

        cmd = [
            sys.executable,
            wav2lip_script,
            "--checkpoint_path", self.wav2lip_checkpoint,
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--face_det_batch_size", str(self.batch_size),
            "--wav2lip_batch_size", str(self.batch_size),
            "--resize_factor", str(self.wav2lip_resize_factor),
            "--fps", str(self.fps)
        ]

        if self.quality == "high":
            cmd.extend(["--nosmooth"])

        logger.debug(f"Running Wav2Lip: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            logger.debug(f"Wav2Lip output: {result.stdout}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Wav2Lip error: {e.stderr}")
            raise RuntimeError(f"Wav2Lip animation failed: {e.stderr}")

    def _post_process_video(self, input_path: str, output_path: str):
        """
        Post-process video to ensure correct format and resolution.

        Args:
            input_path: Path to input video
            output_path: Path to save processed video
        """
        try:
            import ffmpeg

            # Check if input needs processing
            probe = ffmpeg.probe(input_path)
            video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)

            if not video_stream:
                logger.warning(f"No video stream found in {input_path}")
                return

            current_width = int(video_stream['width'])
            current_height = int(video_stream['height'])

            # Skip if already correct resolution
            if current_width == self.output_width and current_height == self.output_height:
                logger.debug("Video already at target resolution")
                if input_path != output_path:
                    import shutil
                    shutil.copy2(input_path, output_path)
                return

            # Resize maintaining aspect ratio with padding
            logger.debug(f"Resizing video from {current_width}x{current_height} to {self.output_width}x{self.output_height}")

            temp_output = Path(output_path).parent / f"temp_{Path(output_path).name}"

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.filter(
                stream,
                'scale',
                self.output_width,
                self.output_height,
                force_original_aspect_ratio='decrease'
            )
            stream = ffmpeg.filter(
                stream,
                'pad',
                self.output_width,
                self.output_height,
                '(ow-iw)/2',
                '(oh-ih)/2',
                color='black'
            )
            stream = ffmpeg.output(
                stream,
                str(temp_output),
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                r=self.fps
            )
            stream = ffmpeg.overwrite_output(stream)

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            # Replace original with processed version
            import shutil
            shutil.move(str(temp_output), output_path)

        except Exception as e:
            logger.error(f"Error post-processing video: {e}")
            # Continue even if post-processing fails
            if input_path != output_path:
                import shutil
                shutil.copy2(input_path, output_path)

    def process_timeline(
        self,
        timeline_path: str,
        image_selections_path: str,
        output_dir: str = "data/animated",
        episode_name: Optional[str] = None,
        resume: bool = True
    ) -> Tuple[List[str], Dict]:
        """
        Process entire timeline and generate animated videos for each line.

        Supports checkpoint/resume functionality for batch processing recovery.
        Progress is saved after each successful animation to allow resuming
        from the last successful point in case of errors.

        Args:
            timeline_path: Path to timeline JSON (from generate_audio.py)
            image_selections_path: Path to image selections JSON (from select_images.py)
            output_dir: Directory to save animated videos
            episode_name: Name for this episode (used in output paths)
            resume: If True, resume from last checkpoint; if False, start fresh

        Returns:
            Tuple of (list of animated video paths, updated timeline dictionary)

        Raises:
            FileNotFoundError: If timeline or selections file not found
            ValueError: If data is invalid or inconsistent
        """
        # Load timeline
        timeline_file = Path(timeline_path)
        if not timeline_file.exists():
            raise FileNotFoundError(f"Timeline file not found: {timeline_path}")

        with open(timeline_file, 'r', encoding='utf-8') as f:
            timeline = json.load(f)

        # Load image selections
        selections_file = Path(image_selections_path)
        if not selections_file.exists():
            raise FileNotFoundError(f"Image selections file not found: {image_selections_path}")

        with open(selections_file, 'r', encoding='utf-8') as f:
            selections = json.load(f)

        # Validate timeline structure
        if not isinstance(timeline, dict):
            raise ValueError("Timeline must be a dictionary")

        if 'episode' not in timeline:
            raise ValueError("Timeline missing 'episode' field")

        if 'lines' not in timeline or not isinstance(timeline['lines'], list):
            raise ValueError("Timeline missing valid 'lines' field")

        if not timeline['lines']:
            raise ValueError("Timeline has no lines")

        # Validate selections structure
        if not isinstance(selections, dict):
            raise ValueError("Selections must be a dictionary")

        if 'selections' not in selections or not isinstance(selections['selections'], list):
            raise ValueError("Selections missing valid 'selections' field")

        # Validate episode consistency
        if timeline['episode'] != selections['episode']:
            raise ValueError(
                f"Episode name mismatch: timeline has '{timeline['episode']}', "
                f"selections has '{selections['episode']}'"
            )

        # Use episode name from timeline if not provided
        if not episode_name:
            episode_name = timeline['episode']

        # Create output directory
        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Checkpoint file for resume functionality
        checkpoint_file = output_path / ".animation_checkpoint.json"

        logger.info(f"Processing timeline for episode: {episode_name}")
        logger.info(f"Lines to animate: {len(timeline['lines'])}")

        # Create mapping of line indices to image paths
        image_map = {}
        for sel in selections['selections']:
            image_map[sel['line_index']] = sel['image_path']

        # Initialize or load checkpoint
        animated_videos = []
        updated_timeline = {
            'episode': episode_name,
            'total_duration': timeline.get('total_duration', 0.0),
            'lines': []
        }
        processed_indices = set()

        if resume and checkpoint_file.exists():
            # Load existing progress
            logger.info(f"Resuming from checkpoint: {checkpoint_file}")
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)

                updated_timeline = checkpoint_data.get('timeline', updated_timeline)
                processed_indices = set(checkpoint_data.get('processed_indices', []))

                # Collect existing animated videos
                for line in updated_timeline['lines']:
                    if line.get('animated_video'):
                        animated_videos.append(line['animated_video'])

                logger.info(f"Resuming: {len(processed_indices)} lines already processed")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
                processed_indices = set()
        else:
            if not resume:
                logger.info("Resume disabled, starting fresh")
            # Clean start - remove checkpoint if it exists
            if checkpoint_file.exists():
                checkpoint_file.unlink()

        # Process each line
        for i, line in enumerate(timeline['lines'], 1):
            line_index = line['index']

            # Skip if already processed (for resume)
            if line_index in processed_indices:
                logger.info(f"[{i}/{len(timeline['lines'])}] Line {line_index} already processed, skipping")
                continue

            try:
                character = line['character']
                audio_file = line['audio_file']

                # Find corresponding image
                if line_index not in image_map:
                    logger.error(f"No image selection found for line {line_index}")
                    raise ValueError(f"Missing image for line {line_index}")

                image_file = image_map[line_index]

                # Generate output filename
                emotion = line.get('emotion', 'neutral')
                output_filename = f"{line_index:03d}_{character.lower()}_{emotion}_animated.mp4"
                output_filepath = output_path / output_filename

                # Animate the image
                logger.info(f"[{i}/{len(timeline['lines'])}] Animating {character}...")

                animated_path = self.animate_image(
                    image_path=image_file,
                    audio_path=audio_file,
                    output_path=str(output_filepath),
                    character=character
                )

                animated_videos.append(animated_path)

                # Update timeline with animated video path
                updated_line = line.copy()
                updated_line['animated_video'] = animated_path
                updated_timeline['lines'].append(updated_line)

                # Mark as processed
                processed_indices.add(line_index)

                # Save checkpoint after each successful animation
                self._save_checkpoint(checkpoint_file, updated_timeline, processed_indices)
                logger.debug(f"Checkpoint saved after line {line_index}")

            except Exception as e:
                logger.error(f"Failed to animate line {i}: {e}")
                # Add line without animated video but continue processing
                updated_line = line.copy()
                updated_line['animated_video'] = None
                updated_line['animation_error'] = str(e)
                updated_timeline['lines'].append(updated_line)

                # Mark as processed (even though it failed) to avoid retry loops
                processed_indices.add(line_index)

                # Save checkpoint to track the error
                self._save_checkpoint(checkpoint_file, updated_timeline, processed_indices)
                continue

        logger.info(f"Successfully animated {len(animated_videos)} out of {len(timeline['lines'])} lines")

        # Remove checkpoint file on successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("Processing complete, checkpoint removed")

        return animated_videos, updated_timeline

    def _save_checkpoint(self, checkpoint_file: Path, timeline: Dict, processed_indices: set):
        """
        Save checkpoint data for resume functionality.

        Args:
            checkpoint_file: Path to checkpoint file
            timeline: Current timeline state
            processed_indices: Set of processed line indices
        """
        try:
            checkpoint_data = {
                'timeline': timeline,
                'processed_indices': list(processed_indices),
                'last_updated': datetime.now().isoformat()
            }

            # Write atomically using a temporary file
            temp_file = checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_file.replace(checkpoint_file)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Don't raise - checkpointing failure shouldn't stop processing

    def save_timeline(self, timeline: Dict, output_dir: str = "data/animated") -> str:
        """
        Save the updated timeline with animated video paths to a JSON file.

        Args:
            timeline: Timeline dictionary with animated video paths
            output_dir: Directory to save the timeline file

        Returns:
            Path to saved timeline file
        """
        episode_name = timeline['episode']
        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)

        timeline_file = output_path / "animated_timeline.json"

        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        logger.info(f"Animated timeline saved to {timeline_file}")
        return str(timeline_file)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate lip-synced animations from static images and audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Animate a single image with audio
  python animate.py \\
    --image data/butcher_images/butcher_001.jpg \\
    --audio data/audio/episode/001_butcher_happy.mp3 \\
    --output data/animated/test.mp4

  # Process entire timeline with SadTalker
  python animate.py \\
    --timeline data/audio/20240101_120000/timeline.json \\
    --images data/image_selections.json \\
    --method sadtalker

  # Process with Wav2Lip and high quality
  python animate.py \\
    --timeline timeline.json \\
    --images selections.json \\
    --method wav2lip \\
    --quality high

Dependencies:
  - FFmpeg (required for all methods)
  - SadTalker: https://github.com/OpenTalker/SadTalker
  - Wav2Lip: https://github.com/Rudrabha/Wav2Lip

Configuration:
  Set the following in config/.env:
  - SADTALKER_CHECKPOINT_PATH: Path to SadTalker models
  - WAV2LIP_CHECKPOINT_PATH: Path to Wav2Lip checkpoint
  - ANIMATION_QUALITY: low, medium, or high (default: high)
  - VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS: Output video settings
        """
    )

    # Mode selection (single image or timeline)
    mode_group = parser.add_mutually_exclusive_group(required=True)

    mode_group.add_argument(
        "--image",
        help="Path to single image file (requires --audio and --output)"
    )

    mode_group.add_argument(
        "--timeline",
        help="Path to timeline JSON file (requires --images)"
    )

    # Single image mode arguments
    parser.add_argument(
        "--audio",
        help="Path to audio file (for single image mode)"
    )

    parser.add_argument(
        "--output",
        help="Output path for animated video (for single image mode)"
    )

    # Timeline mode arguments
    parser.add_argument(
        "--images",
        help="Path to image selections JSON file (for timeline mode)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/animated",
        help="Directory to save animated videos (default: data/animated)"
    )

    parser.add_argument(
        "--episode-name",
        help="Episode name (default: from timeline)"
    )

    # Animation settings
    parser.add_argument(
        "--method",
        choices=["sadtalker", "wav2lip"],
        default="sadtalker",
        help="Animation method to use (default: sadtalker)"
    )

    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        help="Animation quality (overrides env var)"
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume from checkpoint (start fresh)"
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
    if args.image:
        if not args.audio or not args.output:
            parser.error("Single image mode requires --audio and --output")
    elif args.timeline:
        if not args.images:
            parser.error("Timeline mode requires --images")

    # Set quality if provided
    if args.quality:
        os.environ["ANIMATION_QUALITY"] = args.quality

    # Initialize generator
    try:
        generator = AnimationGenerator(method=args.method)
    except Exception as e:
        logger.error(f"Failed to initialize animator: {e}")
        sys.exit(1)

    # Check dependencies
    if not generator._check_dependencies():
        logger.error("Missing required dependencies")
        sys.exit(1)

    # Process based on mode
    try:
        if args.image:
            # Single image mode
            logger.info("Single image animation mode")

            output_path = generator.animate_image(
                image_path=args.image,
                audio_path=args.audio,
                output_path=args.output
            )

            print("\n" + "=" * 80)
            print("ANIMATION COMPLETE")
            print("=" * 80)
            print(f"Animated video: {output_path}")
            print("=" * 80 + "\n")

        else:
            # Timeline mode
            logger.info("Timeline processing mode")

            animated_videos, updated_timeline = generator.process_timeline(
                timeline_path=args.timeline,
                image_selections_path=args.images,
                output_dir=args.output_dir,
                episode_name=args.episode_name,
                resume=not args.no_resume
            )

            # Save updated timeline
            timeline_file = generator.save_timeline(updated_timeline, args.output_dir)

            # Display summary
            episode_name = updated_timeline['episode']
            successful = len([v for v in animated_videos if v])
            total = len(updated_timeline['lines'])

            print("\n" + "=" * 80)
            print("ANIMATION PROCESSING COMPLETE")
            print("=" * 80)
            print(f"Episode: {episode_name}")
            print(f"Method: {args.method}")
            print(f"Videos animated: {successful}/{total}")
            print(f"Output directory: {Path(args.output_dir) / episode_name}")
            print(f"Timeline file: {timeline_file}")
            print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Animation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
