#!/usr/bin/env python3
"""
Generate audio files from script using ElevenLabs TTS.

This script reads a script JSON file and generates audio for each dialogue line
using ElevenLabs API with character-specific voices.
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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


class AudioGenerator:
    """Generates audio files using ElevenLabs TTS."""

    def __init__(self):
        """Initialize the AudioGenerator with ElevenLabs client."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error("ELEVENLABS_API_KEY not found in environment")
            raise ValueError("Please set ELEVENLABS_API_KEY in config/.env")

        # Import ElevenLabs
        try:
            from elevenlabs import ElevenLabs, Voice, VoiceSettings
            self.client = ElevenLabs(api_key=self.api_key)
            self.Voice = Voice
            self.VoiceSettings = VoiceSettings
        except ImportError:
            logger.error("elevenlabs package not installed. Run: pip install elevenlabs")
            sys.exit(1)

        # Voice IDs for characters
        self.voice_ids = {
            "Butcher": os.getenv("ELEVENLABS_VOICE_BUTCHER", "21m00Tcm4TlvDq8ikWAM"),
            "Nutsy": os.getenv("ELEVENLABS_VOICE_NUTSY", "pNInz6obpgDQGcFmaJgB")
        }

        # Voice settings
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")
        self.stability = float(os.getenv("ELEVENLABS_STABILITY", "0.5"))
        self.similarity_boost = float(os.getenv("ELEVENLABS_SIMILARITY_BOOST", "0.75"))
        self.rate_limit = float(os.getenv("ELEVENLABS_RATE_LIMIT", "3"))

        logger.info("ElevenLabs client initialized")
        logger.info(f"Butcher voice ID: {self.voice_ids['Butcher']}")
        logger.info(f"Nutsy voice ID: {self.voice_ids['Nutsy']}")

    def generate_audio(
        self,
        text: str,
        character: str,
        emotion: str = "neutral"
    ) -> bytes:
        """
        Generate audio for a single line of dialogue.

        Args:
            text: The text to convert to speech
            character: The character speaking (Butcher or Nutsy)
            emotion: The emotion to convey

        Returns:
            Audio data as bytes
        """
        # Validate text input
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # ElevenLabs API has a 5000 character limit per request
        if len(text) > 5000:
            raise ValueError(f"Text too long ({len(text)} chars). Maximum is 5000 characters.")

        if character not in self.voice_ids:
            raise ValueError(f"Unknown character: {character}")

        voice_id = self.voice_ids[character]

        try:
            logger.debug(f"Generating audio for {character}: {text[:50]}...")

            # Adjust voice settings based on emotion
            settings = self._get_voice_settings(emotion)

            # Generate audio
            audio = self.client.generate(
                text=text,
                voice=self.Voice(
                    voice_id=voice_id,
                    settings=settings
                ),
                model=self.model
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio)

            logger.debug(f"Generated {len(audio_bytes)} bytes of audio")
            return audio_bytes

        except Exception as e:
            logger.error(f"Error generating audio for {character}: {e}")
            raise

    def _get_voice_settings(self, emotion: str) -> 'VoiceSettings':
        """
        Get voice settings adjusted for the given emotion.

        Args:
            emotion: The emotion to convey

        Returns:
            VoiceSettings object
        """
        # Base settings
        stability = self.stability
        similarity_boost = self.similarity_boost

        # Adjust based on emotion
        emotion_adjustments = {
            "excited": {"stability": stability - 0.1, "similarity_boost": similarity_boost + 0.05},
            "angry": {"stability": stability - 0.15, "similarity_boost": similarity_boost + 0.1},
            "sad": {"stability": stability + 0.1, "similarity_boost": similarity_boost - 0.05},
            "sarcastic": {"stability": stability, "similarity_boost": similarity_boost},
            "confused": {"stability": stability - 0.05, "similarity_boost": similarity_boost},
            "surprised": {"stability": stability - 0.1, "similarity_boost": similarity_boost + 0.05},
            "happy": {"stability": stability - 0.05, "similarity_boost": similarity_boost + 0.05},
            "neutral": {"stability": stability, "similarity_boost": similarity_boost}
        }

        settings = emotion_adjustments.get(emotion.lower(), emotion_adjustments["neutral"])

        # Clamp values to valid range
        stability = max(0.0, min(1.0, settings["stability"]))
        similarity_boost = max(0.0, min(1.0, settings["similarity_boost"]))

        return self.VoiceSettings(
            stability=stability,
            similarity_boost=similarity_boost
        )

    def save_audio(self, audio_bytes: bytes, filepath: Path):
        """
        Save audio bytes to a file.

        Args:
            audio_bytes: The audio data
            filepath: Path to save the file
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            f.write(audio_bytes)

        logger.debug(f"Saved audio to {filepath}")

    def get_audio_duration(self, filepath: Path) -> float:
        """
        Get the duration of an audio file in seconds.

        Args:
            filepath: Path to the audio file

        Returns:
            Duration in seconds
        """
        try:
            import soundfile as sf
            data, samplerate = sf.read(filepath)
            duration = len(data) / samplerate
            return duration
        except ImportError:
            logger.warning("soundfile not installed, cannot determine audio duration")
            logger.info("Run: pip install soundfile")
            return 0.0
        except Exception as e:
            logger.error(f"Error reading audio file {filepath}: {e}")
            return 0.0

    def _generate_single_audio(
        self,
        line_data: Tuple[int, Dict[str, str], Path]
    ) -> Optional[Dict]:
        """
        Generate audio for a single line (used by parallel processing).

        Args:
            line_data: Tuple of (index, line_dict, output_path)

        Returns:
            Dictionary with line info and audio file path, or None if failed
        """
        i, line, output_path = line_data
        character = line['character']
        text = line['line']
        emotion = line.get('emotion', 'neutral')

        # Generate filename
        filename = f"{i:03d}_{character.lower()}_{emotion}.mp3"
        filepath = output_path / filename

        try:
            logger.info(f"[{i}] Generating audio for {character}...")

            max_retries = 3
            base_delay = 2.0
            audio_bytes = None

            for retry in range(max_retries):
                try:
                    audio_bytes = self.generate_audio(text, character, emotion)
                    break  # Success, exit retry loop
                except Exception as e:
                    # Check if it's a rate limit error
                    error_str = str(e).lower()
                    is_rate_limit = any(phrase in error_str for phrase in ['rate limit', 'too many requests', '429'])

                    if is_rate_limit and retry < max_retries - 1:
                        wait_time = (2 ** retry) * base_delay
                        logger.warning(f"Rate limit hit for line {i}, retrying in {wait_time:.1f}s (attempt {retry + 1}/{max_retries})...")
                        time.sleep(wait_time)
                    else:
                        # Not a rate limit error or final retry, re-raise
                        raise

            if audio_bytes is None:
                raise Exception("Failed to generate audio after retries")

            self.save_audio(audio_bytes, filepath)

            # Get duration
            duration = self.get_audio_duration(filepath)

            return {
                'index': i,
                'character': character,
                'text': text,
                'emotion': emotion,
                'audio_file': str(filepath),
                'duration': duration
            }

        except Exception as e:
            logger.error(f"Failed to generate audio for line {i}: {e}")
            return None

    def process_script(
        self,
        script: List[Dict[str, str]],
        output_dir: str = "data/audio",
        episode_name: str = None,
        parallel: bool = False,
        max_workers: int = None
    ) -> Tuple[List[str], Dict]:
        """
        Process an entire script and generate audio files.

        Args:
            script: List of dialogue line dictionaries
            output_dir: Directory to save audio files
            episode_name: Name for this episode (used in filenames)
            parallel: Enable parallel processing
            max_workers: Number of parallel workers (default from env)

        Returns:
            Tuple of (list of audio file paths, timeline dictionary)
        """
        if not episode_name:
            from datetime import datetime
            episode_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)

        if parallel:
            return self._process_script_parallel(script, output_path, episode_name, max_workers)
        else:
            return self._process_script_sequential(script, output_path, episode_name)

    def _process_script_sequential(
        self,
        script: List[Dict[str, str]],
        output_path: Path,
        episode_name: str
    ) -> Tuple[List[str], Dict]:
        """Process script sequentially (original behavior)."""
        audio_files = []
        timeline = {
            'episode': episode_name,
            'total_duration': 0.0,
            'lines': []
        }

        current_time = 0.0

        for i, line in enumerate(script, 1):
            character = line['character']
            text = line['line']
            emotion = line.get('emotion', 'neutral')

            # Generate filename
            filename = f"{i:03d}_{character.lower()}_{emotion}.mp3"
            filepath = output_path / filename

            try:
                # Generate audio with retry logic and exponential backoff
                logger.info(f"[{i}/{len(script)}] Generating audio for {character}...")

                max_retries = 3
                base_delay = 2.0
                audio_bytes = None

                for retry in range(max_retries):
                    try:
                        audio_bytes = self.generate_audio(text, character, emotion)
                        break  # Success, exit retry loop
                    except Exception as e:
                        # Check if it's a rate limit error (ElevenLabs raises different exceptions)
                        error_str = str(e).lower()
                        is_rate_limit = any(phrase in error_str for phrase in ['rate limit', 'too many requests', '429'])

                        if is_rate_limit and retry < max_retries - 1:
                            wait_time = (2 ** retry) * base_delay
                            logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s (attempt {retry + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            # Not a rate limit error or final retry, re-raise
                            raise

                if audio_bytes is None:
                    raise Exception("Failed to generate audio after retries")

                self.save_audio(audio_bytes, filepath)

                # Get duration
                duration = self.get_audio_duration(filepath)

                # Record in timeline
                audio_files.append(str(filepath))
                timeline['lines'].append({
                    'index': i,
                    'character': character,
                    'text': text,
                    'emotion': emotion,
                    'audio_file': str(filepath),
                    'start_time': current_time,
                    'duration': duration,
                    'end_time': current_time + duration
                })

                current_time += duration

                # Rate limiting
                if i < len(script):  # Don't sleep after the last line
                    sleep_time = 1.0 / self.rate_limit
                    logger.debug(f"Sleeping {sleep_time:.2f}s for rate limiting...")
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Failed to generate audio for line {i}: {e}")
                # Continue with next line rather than failing completely
                continue

        timeline['total_duration'] = current_time
        logger.info(f"Generated {len(audio_files)} audio files")
        logger.info(f"Total duration: {current_time:.2f} seconds")

        return audio_files, timeline

    def _process_script_parallel(
        self,
        script: List[Dict[str, str]],
        output_path: Path,
        episode_name: str,
        max_workers: int = None
    ) -> Tuple[List[str], Dict]:
        """Process script with parallel audio generation."""
        if max_workers is None:
            max_workers = int(os.getenv("PARALLEL_AUDIO_WORKERS", "3"))

        logger.info(f"Using parallel processing with {max_workers} workers")

        # Prepare work items
        work_items = [
            (i, line, output_path)
            for i, line in enumerate(script, 1)
        ]

        # Process in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._generate_single_audio, item): item[0]
                for item in work_items
            }

            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Completed line {index}/{len(script)}")
                except Exception as e:
                    logger.error(f"Exception processing line {index}: {e}")

        # Sort results by index
        results.sort(key=lambda x: x['index'])

        # Build timeline
        timeline = {
            'episode': episode_name,
            'total_duration': 0.0,
            'lines': []
        }

        audio_files = []
        current_time = 0.0

        for result in results:
            audio_files.append(result['audio_file'])
            timeline['lines'].append({
                'index': result['index'],
                'character': result['character'],
                'text': result['text'],
                'emotion': result['emotion'],
                'audio_file': result['audio_file'],
                'start_time': current_time,
                'duration': result['duration'],
                'end_time': current_time + result['duration']
            })
            current_time += result['duration']

        timeline['total_duration'] = current_time
        logger.info(f"Generated {len(audio_files)} audio files (parallel)")
        logger.info(f"Total duration: {current_time:.2f} seconds")

        return audio_files, timeline

    def save_timeline(self, timeline: Dict, output_dir: str = "data/audio"):
        """
        Save the timeline data to a JSON file.

        Args:
            timeline: Timeline dictionary
            output_dir: Directory to save the timeline file
        """
        episode_name = timeline['episode']
        output_path = Path(output_dir) / episode_name
        output_path.mkdir(parents=True, exist_ok=True)

        timeline_file = output_path / "timeline.json"

        with open(timeline_file, 'w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)

        logger.info(f"Timeline saved to {timeline_file}")
        return str(timeline_file)


def load_script(script_path: str) -> List[Dict[str, str]]:
    """
    Load a script from a JSON file.

    Args:
        script_path: Path to the script JSON file

    Returns:
        List of dialogue line dictionaries
    """
    script_file = Path(script_path)

    if not script_file.exists():
        raise FileNotFoundError(f"Script file not found: {script_path}")

    with open(script_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both direct script arrays and wrapped format
    if isinstance(data, list):
        script = data
    elif isinstance(data, dict) and 'script' in data:
        script = data['script']
    else:
        raise ValueError("Invalid script format")

    logger.info(f"Loaded script with {len(script)} lines from {script_path}")
    return script


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate audio files from dialogue script using ElevenLabs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate audio from a script file
  python generate_audio.py data/scripts/episode_20240101_120000.json

  # Generate with custom output directory
  python generate_audio.py script.json --output-dir data/audio/test

  # Generate with custom episode name
  python generate_audio.py script.json --episode-name pilot_episode
        """
    )

    parser.add_argument(
        "script_file",
        help="Path to the script JSON file"
    )

    parser.add_argument(
        "--output-dir",
        default="data/audio",
        help="Directory to save audio files (default: data/audio)"
    )

    parser.add_argument(
        "--episode-name",
        help="Name for this episode (default: timestamp from script filename or current time)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel audio generation"
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Number of parallel workers (default: from PARALLEL_AUDIO_WORKERS env var)"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine episode name
    episode_name = args.episode_name
    if not episode_name:
        # Try to extract from script filename
        script_path = Path(args.script_file)
        if script_path.stem.startswith("episode_"):
            episode_name = script_path.stem.replace("episode_", "")
        else:
            from datetime import datetime
            episode_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Episode name: {episode_name}")

    # Load script
    try:
        script = load_script(args.script_file)
    except Exception as e:
        logger.error(f"Failed to load script: {e}")
        sys.exit(1)

    # Generate audio
    try:
        generator = AudioGenerator()
        audio_files, timeline = generator.process_script(
            script,
            output_dir=args.output_dir,
            episode_name=episode_name,
            parallel=args.parallel,
            max_workers=args.workers
        )

        # Save timeline
        timeline_file = generator.save_timeline(timeline, args.output_dir)

        # Display summary
        print("\n" + "=" * 80)
        print("AUDIO GENERATION COMPLETE")
        print("=" * 80)
        print(f"Episode: {episode_name}")
        print(f"Audio files generated: {len(audio_files)}")
        print(f"Total duration: {timeline['total_duration']:.2f} seconds")
        print(f"Output directory: {Path(args.output_dir) / episode_name}")
        print(f"Timeline file: {timeline_file}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
