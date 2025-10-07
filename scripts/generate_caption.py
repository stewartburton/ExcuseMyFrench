#!/usr/bin/env python3
"""
Generate engaging Instagram captions from video content.

This script analyzes video scripts and generates captions that:
- Capture the essence of the video
- Include trending topics
- Add relevant hashtags
- Maintain character voice (Butcher's personality)
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

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


class CaptionGenerator:
    """Generates Instagram captions from video content."""

    def __init__(self):
        """Initialize the CaptionGenerator."""
        # Load configuration
        self.default_hashtags = os.getenv(
            "INSTAGRAM_HASHTAGS",
            "#frenchbulldog #comedy #funny #reels #dogsofinstagram #fyp"
        )
        self.caption_template = os.getenv(
            "INSTAGRAM_CAPTION_TEMPLATE",
            "{hook}\n\n{content}\n\n{hashtags}"
        )

        # Initialize LLM client if available
        self.llm_client = self._init_llm_client()

        logger.info("CaptionGenerator initialized")

    def _init_llm_client(self):
        """Initialize LLM client for caption generation."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Try Anthropic first
        if anthropic_key and anthropic_key.startswith("sk-ant-"):
            try:
                import anthropic
                logger.info("Using Anthropic for caption generation")
                return anthropic.Anthropic(api_key=anthropic_key)
            except ImportError:
                logger.warning("anthropic package not installed")

        # Fall back to OpenAI
        if openai_key and openai_key.startswith("sk-"):
            try:
                import openai
                logger.info("Using OpenAI for caption generation")
                return openai.OpenAI(api_key=openai_key)
            except ImportError:
                logger.warning("openai package not installed")

        logger.warning("No LLM client available, using template-based captions")
        return None

    def _extract_script_from_video(self, video_path: str) -> Optional[List[dict]]:
        """
        Extract script from video metadata or associated files.

        Args:
            video_path: Path to video file

        Returns:
            List of script lines or None
        """
        video_file = Path(video_path)

        # Look for associated script file
        # Format: data/scripts/episode_TIMESTAMP.json
        scripts_dir = Path("data") / "scripts"
        if not scripts_dir.exists():
            logger.warning(f"Scripts directory not found: {scripts_dir}")
            return None

        # Try to find matching script by episode name
        episode_name = video_file.stem
        for script_file in scripts_dir.glob("*.json"):
            try:
                with open(script_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if this script matches
                    if 'script' in data:
                        return data['script']
            except Exception as e:
                logger.debug(f"Error reading {script_file}: {e}")
                continue

        return None

    def _generate_hook(self, script_lines: List[dict]) -> str:
        """
        Generate an attention-grabbing hook from the script.

        Args:
            script_lines: List of script line dictionaries

        Returns:
            Hook text
        """
        if not script_lines:
            return "You won't believe what happens next!"

        # Use the first line or find an exciting moment
        first_line = script_lines[0]['line'] if script_lines else ""

        # Look for lines with high-energy emotions
        exciting_emotions = ['excited', 'surprised', 'angry']
        for line in script_lines[:3]:  # Check first 3 lines
            if line.get('emotion') in exciting_emotions:
                first_line = line['line']
                break

        # Truncate and add intrigue
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."

        return first_line

    def _extract_topics(self, script_lines: List[dict]) -> List[str]:
        """
        Extract trending topics from script.

        Args:
            script_lines: List of script line dictionaries

        Returns:
            List of topics
        """
        topics = []

        # Common topic keywords to look for
        topic_keywords = [
            'AI', 'artificial intelligence', 'climate', 'crypto', 'bitcoin',
            'politics', 'election', 'social media', 'tiktok', 'instagram',
            'technology', 'science', 'space', 'mars', 'elon musk',
            'celebrity', 'movies', 'music', 'sports', 'gaming'
        ]

        # Extract topics from dialogue
        for line in script_lines:
            text = line['line'].lower()
            for keyword in topic_keywords:
                if keyword.lower() in text and keyword not in topics:
                    topics.append(keyword)

        return topics[:3]  # Return top 3

    def _generate_hashtags(self, topics: List[str]) -> str:
        """
        Generate relevant hashtags from topics.

        Args:
            topics: List of topic strings

        Returns:
            Hashtag string
        """
        hashtags = []

        # Add topic-based hashtags
        for topic in topics:
            # Clean and format topic as hashtag
            tag = re.sub(r'[^\w\s]', '', topic)
            tag = tag.replace(' ', '').lower()
            if tag and f"#{tag}" not in hashtags:
                hashtags.append(f"#{tag}")

        # Add default hashtags
        default_tags = self.default_hashtags.split()
        for tag in default_tags:
            if tag not in hashtags:
                hashtags.append(tag)

        return " ".join(hashtags[:15])  # Instagram allows up to 30, use 15

    def generate_with_llm(self, script_lines: List[dict]) -> str:
        """
        Generate caption using LLM.

        Args:
            script_lines: List of script line dictionaries

        Returns:
            Generated caption
        """
        if not self.llm_client:
            return self._generate_template_caption(script_lines)

        # Build prompt
        script_text = "\n".join([
            f"{line['character']}: {line['line']}"
            for line in script_lines
        ])

        prompt = f"""Generate an engaging Instagram caption for this comedy video featuring Butcher (a sarcastic French bulldog) and Nutsy (a hyperactive squirrel).

Script:
{script_text}

Requirements:
- Start with an attention-grabbing hook (1-2 sentences)
- Keep it concise and engaging (2-3 sentences total)
- Match Butcher's sarcastic, witty personality
- Make people want to watch the video
- Don't include hashtags (we'll add those separately)
- Use emojis sparingly (1-2 max)

Caption:"""

        try:
            # Try Anthropic
            if hasattr(self.llm_client, 'messages'):
                response = self.llm_client.messages.create(
                    model=os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022"),
                    max_tokens=200,
                    temperature=0.8,
                    messages=[{"role": "user", "content": prompt}]
                )
                caption = response.content[0].text.strip()
            # Try OpenAI
            elif hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model=os.getenv("LLM_MODEL", "gpt-4"),
                    max_tokens=200,
                    temperature=0.8,
                    messages=[
                        {"role": "system", "content": "You are a social media caption writer."},
                        {"role": "user", "content": prompt}
                    ]
                )
                caption = response.choices[0].message.content.strip()
            else:
                return self._generate_template_caption(script_lines)

            logger.info("Generated caption with LLM")
            return caption

        except Exception as e:
            logger.warning(f"LLM caption generation failed: {e}")
            return self._generate_template_caption(script_lines)

    def _generate_template_caption(self, script_lines: List[dict]) -> str:
        """
        Generate caption using template (fallback).

        Args:
            script_lines: List of script line dictionaries

        Returns:
            Generated caption
        """
        hook = self._generate_hook(script_lines)
        topics = self._extract_topics(script_lines)

        # Create content based on topics
        if topics:
            content = f"Butcher and Nutsy tackle {', '.join(topics)} in their own special way."
        else:
            content = "Butcher and Nutsy are at it again with their hilarious banter!"

        caption = f"{hook}\n\n{content}"
        logger.info("Generated caption with template")
        return caption

    def generate_from_script(self, script_lines: List[dict]) -> str:
        """
        Generate complete caption from script.

        Args:
            script_lines: List of script line dictionaries

        Returns:
            Complete caption with hashtags
        """
        if not script_lines:
            logger.warning("No script lines provided")
            return "Watch this hilarious moment!\n\n" + self.default_hashtags

        # Generate main caption
        if self.llm_client:
            caption = self.generate_with_llm(script_lines)
        else:
            caption = self._generate_template_caption(script_lines)

        # Add hashtags
        topics = self._extract_topics(script_lines)
        hashtags = self._generate_hashtags(topics)

        # Combine
        full_caption = f"{caption}\n\n{hashtags}"

        # Ensure caption is within Instagram's limit (2,200 characters)
        if len(full_caption) > 2200:
            # Truncate caption but keep hashtags
            max_caption_length = 2200 - len(hashtags) - 10
            caption = caption[:max_caption_length] + "..."
            full_caption = f"{caption}\n\n{hashtags}"

        return full_caption

    def generate_from_video(self, video_path: str) -> str:
        """
        Generate caption from video file.

        Args:
            video_path: Path to video file

        Returns:
            Generated caption
        """
        logger.info(f"Generating caption for: {video_path}")

        # Try to extract script
        script_lines = self._extract_script_from_video(video_path)

        if script_lines:
            logger.info(f"Found script with {len(script_lines)} lines")
            return self.generate_from_script(script_lines)
        else:
            logger.warning("No script found, using generic caption")
            video_name = Path(video_path).stem
            return f"New episode: {video_name}\n\nButcher and Nutsy are back with more hilarious moments!\n\n{self.default_hashtags}"

    def generate_from_script_file(self, script_path: str) -> str:
        """
        Generate caption from script file.

        Args:
            script_path: Path to script JSON file

        Returns:
            Generated caption
        """
        logger.info(f"Generating caption from script: {script_path}")

        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'script' not in data:
                raise ValueError("No 'script' field in JSON file")

            return self.generate_from_script(data['script'])

        except Exception as e:
            logger.error(f"Error reading script file: {e}")
            return f"Watch this hilarious moment!\n\n{self.default_hashtags}"


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate Instagram captions from video content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate caption from video file
  python generate_caption.py data/final_videos/episode_20240101.mp4

  # Generate caption from script file
  python generate_caption.py --script data/scripts/episode_20240101.json

  # Generate with custom hashtags
  python generate_caption.py video.mp4 --hashtags "#custom #tags"

  # Save caption to file
  python generate_caption.py video.mp4 --output caption.txt
        """
    )

    parser.add_argument(
        "input",
        help="Path to video file or script JSON"
    )

    parser.add_argument(
        "--script",
        action="store_true",
        help="Input is a script JSON file (not a video)"
    )

    parser.add_argument(
        "--hashtags",
        help="Custom hashtags to use instead of defaults"
    )

    parser.add_argument(
        "--output",
        help="Save caption to file (instead of printing)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate input
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Initialize generator
    generator = CaptionGenerator()

    # Override default hashtags if provided
    if args.hashtags:
        generator.default_hashtags = args.hashtags

    # Generate caption
    try:
        if args.script:
            caption = generator.generate_from_script_file(args.input)
        else:
            caption = generator.generate_from_video(args.input)

        # Output
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(caption)
            print(f"Caption saved to: {args.output}")
        else:
            print("\n" + "=" * 80)
            print("GENERATED CAPTION")
            print("=" * 80)
            print(caption)
            print("=" * 80)
            print(f"\nLength: {len(caption)} characters")
            print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"Failed to generate caption: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
