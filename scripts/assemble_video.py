#!/usr/bin/env python3
"""
Assemble final video from audio, images, and optional background music.

This script uses ffmpeg to create a video with:
- Character images for each dialogue line
- Synchronized audio
- Optional background music at low volume
- Burned-in subtitles
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Optional

import ffmpeg
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


class VideoAssembler:
    """Assembles final video from components."""

    def __init__(self):
        """Initialize the VideoAssembler."""
        # Video settings
        self.width = int(os.getenv("VIDEO_WIDTH", "1080"))
        self.height = int(os.getenv("VIDEO_HEIGHT", "1920"))
        self.fps = int(os.getenv("VIDEO_FPS", "30"))

        # Audio settings
        self.music_volume = float(os.getenv("MUSIC_VOLUME", "0.2"))

        logger.info(f"Video settings: {self.width}x{self.height} @ {self.fps}fps")

    def create_image_clip(
        self,
        image_path: str,
        duration: float,
        output_path: str
    ):
        """
        Create a video clip from a static image.

        Args:
            image_path: Path to the image file
            duration: Duration in seconds
            output_path: Path to save the clip
        """
        try:
            logger.debug(f"Creating clip: {image_path} ({duration:.2f}s)")

            stream = ffmpeg.input(image_path, loop=1, t=duration)
            stream = ffmpeg.filter(stream, 'scale', self.width, self.height, force_original_aspect_ratio='decrease')
            stream = ffmpeg.filter(stream, 'pad', self.width, self.height, '(ow-iw)/2', '(oh-ih)/2', color='black')
            stream = ffmpeg.output(stream, output_path, vcodec='libx264', pix_fmt='yuv420p', r=self.fps)
            stream = ffmpeg.overwrite_output(stream)

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
            logger.debug(f"Clip created: {output_path}")

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error creating clip: {e.stderr.decode()}")
            raise

    def concatenate_clips(
        self,
        clip_list: List[str],
        output_path: str
    ):
        """
        Concatenate multiple video clips.

        Args:
            clip_list: List of paths to video clips
            output_path: Path to save concatenated video
        """
        try:
            logger.info(f"Concatenating {len(clip_list)} clips...")

            # Create concat file
            concat_file = Path(output_path).parent / "concat_list.txt"
            with open(concat_file, 'w') as f:
                for clip in clip_list:
                    f.write(f"file '{Path(clip).absolute()}'\n")

            # Concatenate using concat demuxer
            stream = ffmpeg.input(str(concat_file), format='concat', safe=0)
            stream = ffmpeg.output(stream, output_path, c='copy')
            stream = ffmpeg.overwrite_output(stream)

            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            # Clean up concat file
            concat_file.unlink()

            logger.info(f"Clips concatenated: {output_path}")

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error concatenating clips: {e.stderr.decode()}")
            raise

    def add_audio(
        self,
        video_path: str,
        audio_files: List[str],
        timeline: Dict,
        output_path: str
    ):
        """
        Add audio track to video.

        Args:
            video_path: Path to video file
            audio_files: List of audio file paths (in order)
            timeline: Timeline dictionary with timing information
            output_path: Path to save output video
        """
        try:
            logger.info("Adding audio track...")

            # Create audio concat list
            audio_concat_file = Path(output_path).parent / "audio_concat.txt"
            with open(audio_concat_file, 'w') as f:
                for audio_file in audio_files:
                    f.write(f"file '{Path(audio_file).absolute()}'\n")

            # Concatenate audio files
            temp_audio = Path(output_path).parent / "temp_audio.mp3"
            audio_stream = ffmpeg.input(str(audio_concat_file), format='concat', safe=0)
            audio_stream = ffmpeg.output(audio_stream, str(temp_audio), acodec='libmp3lame')
            audio_stream = ffmpeg.overwrite_output(audio_stream)
            ffmpeg.run(audio_stream, capture_stdout=True, capture_stderr=True, quiet=True)

            # Combine video and audio
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(str(temp_audio))

            stream = ffmpeg.output(
                video,
                audio,
                output_path,
                vcodec='copy',
                acodec='aac',
                shortest=None
            )
            stream = ffmpeg.overwrite_output(stream)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            # Clean up temp files
            audio_concat_file.unlink()
            temp_audio.unlink()

            logger.info(f"Audio added: {output_path}")

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error adding audio: {e.stderr.decode()}")
            raise

    def add_background_music(
        self,
        video_path: str,
        music_path: str,
        output_path: str,
        volume: float = None
    ):
        """
        Add background music to video.

        Args:
            video_path: Path to video file
            music_path: Path to music file
            output_path: Path to save output video
            volume: Music volume (0.0 - 1.0), uses default if None
        """
        if volume is None:
            volume = self.music_volume

        try:
            logger.info(f"Adding background music (volume: {volume})...")

            video = ffmpeg.input(video_path)
            music = ffmpeg.input(music_path, stream_loop=-1)  # Loop music

            # Mix audio: dialogue at 100%, music at specified volume
            audio_mix = ffmpeg.filter(
                [video.audio, music],
                'amix',
                inputs=2,
                duration='first',
                weights=f'1 {volume}'
            )

            stream = ffmpeg.output(
                video.video,
                audio_mix,
                output_path,
                vcodec='copy',
                acodec='aac'
            )
            stream = ffmpeg.overwrite_output(stream)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            logger.info(f"Background music added: {output_path}")

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error adding music: {e.stderr.decode()}")
            raise

    def add_subtitles(
        self,
        video_path: str,
        timeline: Dict,
        output_path: str,
        font_size: int = 48,
        font_color: str = "white",
        outline_color: str = "black"
    ):
        """
        Burn subtitles into video.

        Args:
            video_path: Path to video file
            timeline: Timeline dictionary with dialogue and timing
            output_path: Path to save output video
            font_size: Font size for subtitles
            font_color: Color of subtitle text
            outline_color: Color of text outline
        """
        try:
            logger.info("Adding subtitles...")

            # Create SRT subtitle file
            srt_file = Path(output_path).parent / "subtitles.srt"
            self._create_srt(timeline, srt_file)

            # Burn subtitles using drawtext filter
            # Note: Using ass/srt directly requires libass
            video = ffmpeg.input(video_path)

            # Use subtitles filter for SRT files
            stream = ffmpeg.filter(
                video,
                'subtitles',
                filename=str(srt_file.absolute()).replace('\\', '/').replace(':', '\\:'),
                force_style=f"FontSize={font_size},PrimaryColour=&H{self._color_to_ass(font_color)},OutlineColour=&H{self._color_to_ass(outline_color)},Outline=2"
            )

            stream = ffmpeg.output(stream, output_path, acodec='copy')
            stream = ffmpeg.overwrite_output(stream)
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)

            logger.info(f"Subtitles added: {output_path}")

            # Keep SRT file for reference
            final_srt = Path(output_path).parent / f"{Path(output_path).stem}.srt"
            srt_file.rename(final_srt)

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error adding subtitles: {e.stderr.decode()}")
            logger.warning("Subtitle addition failed, continuing without subtitles")
            # Copy video as-is if subtitle fails
            import shutil
            shutil.copy2(video_path, output_path)

    def _create_srt(self, timeline: Dict, output_path: Path):
        """Create SRT subtitle file from timeline."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(timeline['lines'], 1):
                start_time = self._format_srt_time(line['start_time'])
                end_time = self._format_srt_time(line['end_time'])

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{line['character']}: {line['text']}\n")
                f.write("\n")

    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds to SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def _color_to_ass(self, color: str) -> str:
        """Convert color name to ASS format."""
        colors = {
            'white': 'FFFFFF',
            'black': '000000',
            'yellow': 'FFFF00',
            'cyan': '00FFFF',
            'red': 'FF0000',
            'blue': '0000FF'
        }
        return colors.get(color.lower(), 'FFFFFF')

    def assemble_video(
        self,
        timeline_path: str,
        image_selections_path: str,
        output_dir: str = "data/final_videos",
        music_path: Optional[str] = None,
        add_subtitles: bool = True
    ) -> str:
        """
        Assemble complete video from all components.

        Args:
            timeline_path: Path to timeline JSON (from generate_audio.py)
            image_selections_path: Path to image selections JSON (from select_images.py)
            output_dir: Directory to save final video
            music_path: Optional path to background music file
            add_subtitles: Whether to add subtitles

        Returns:
            Path to final video file
        """
        # Load timeline
        with open(timeline_path, 'r') as f:
            timeline = json.load(f)

        # Load image selections
        with open(image_selections_path, 'r') as f:
            selections = json.load(f)

        episode_name = timeline['episode']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        temp_dir = output_path / f"temp_{episode_name}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Assembling video: {episode_name}")

        try:
            # Step 1: Create video clips for each line
            clip_files = []
            audio_files = []

            for i, line in enumerate(timeline['lines']):
                # Find matching image
                image_path = None
                for sel in selections['selections']:
                    if sel['line_index'] == line['index']:
                        image_path = sel['image_path']
                        break

                if not image_path or not Path(image_path).exists():
                    logger.error(f"Image not found for line {i+1}")
                    raise FileNotFoundError(f"Missing image for line {i+1}")

                # Create clip
                clip_path = temp_dir / f"clip_{i+1:03d}.mp4"
                self.create_image_clip(image_path, line['duration'], str(clip_path))
                clip_files.append(str(clip_path))
                audio_files.append(line['audio_file'])

            # Step 2: Concatenate video clips
            video_no_audio = temp_dir / "video_no_audio.mp4"
            self.concatenate_clips(clip_files, str(video_no_audio))

            # Step 3: Add audio track
            video_with_audio = temp_dir / "video_with_audio.mp4"
            self.add_audio(str(video_no_audio), audio_files, timeline, str(video_with_audio))

            # Step 4: Add background music (if provided)
            if music_path and Path(music_path).exists():
                video_with_music = temp_dir / "video_with_music.mp4"
                self.add_background_music(str(video_with_audio), music_path, str(video_with_music))
                current_video = video_with_music
            else:
                current_video = video_with_audio

            # Step 5: Add subtitles (if requested)
            if add_subtitles:
                final_video = output_path / f"{episode_name}.mp4"
                self.add_subtitles(str(current_video), timeline, str(final_video))
            else:
                final_video = output_path / f"{episode_name}.mp4"
                import shutil
                shutil.copy2(str(current_video), str(final_video))

            logger.info(f"Video assembly complete: {final_video}")

            # Clean up temp files if requested
            if not os.getenv("SAVE_INTERMEDIATE_FILES", "true").lower() == "true":
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")

            return str(final_video)

        except Exception as e:
            logger.error(f"Error during video assembly: {e}")
            raise


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Assemble final video from audio and images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assemble video from timeline and image selections
  python assemble_video.py \\
    --timeline data/audio/20240101_120000/timeline.json \\
    --images data/image_selections.json

  # Add background music
  python assemble_video.py \\
    --timeline data/audio/episode/timeline.json \\
    --images selections.json \\
    --music data/music/background.mp3

  # Skip subtitles
  python assemble_video.py \\
    --timeline timeline.json \\
    --images selections.json \\
    --no-subtitles
        """
    )

    parser.add_argument(
        "--timeline",
        required=True,
        help="Path to timeline JSON file (from generate_audio.py)"
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Path to image selections JSON file (from select_images.py)"
    )

    parser.add_argument(
        "--music",
        help="Path to background music file (optional)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/final_videos",
        help="Directory to save final video (default: data/final_videos)"
    )

    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Don't add subtitles to video"
    )

    parser.add_argument(
        "--music-volume",
        type=float,
        help="Background music volume (0.0 - 1.0)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found. Please install FFmpeg and add it to PATH")
        sys.exit(1)

    # Initialize assembler
    assembler = VideoAssembler()

    if args.music_volume is not None:
        assembler.music_volume = args.music_volume

    # Assemble video
    try:
        final_video = assembler.assemble_video(
            timeline_path=args.timeline,
            image_selections_path=args.images,
            output_dir=args.output_dir,
            music_path=args.music,
            add_subtitles=not args.no_subtitles
        )

        print("\n" + "=" * 80)
        print("VIDEO ASSEMBLY COMPLETE")
        print("=" * 80)
        print(f"Final video: {final_video}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Failed to assemble video: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
