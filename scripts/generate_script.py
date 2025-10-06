#!/usr/bin/env python3
"""
Generate dialogue scripts for Butcher and Nutsy using LLM.

This script reads trending topics from the database and uses an LLM
(Anthropic Claude or OpenAI GPT) to generate humorous dialogue between
the two characters.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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


class ScriptGenerator:
    """Generates dialogue scripts using LLM."""

    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the ScriptGenerator.

        Args:
            llm_provider: LLM provider to use ('anthropic' or 'openai').
                         If None, auto-detect based on available API keys.
        """
        self.llm_provider = llm_provider or self._detect_provider()
        self.client = self._init_client()

        # Character configurations
        self.butcher_personality = os.getenv(
            "BUTCHER_PERSONALITY",
            "sarcastic, wise, French bulldog, deep voice, witty, observant"
        )
        self.nutsy_personality = os.getenv(
            "NUTSY_PERSONALITY",
            "hyperactive, energetic, squirrel, high-pitched voice, excitable, naive"
        )

        # Script settings
        self.min_lines = int(os.getenv("SCRIPT_LINES_MIN", "4"))
        self.max_lines = int(os.getenv("SCRIPT_LINES_MAX", "12"))
        self.content_tone = os.getenv("CONTENT_TONE", "humorous, sarcastic, topical")

        logger.info(f"Using LLM provider: {self.llm_provider}")

    def _detect_provider(self) -> str:
        """Auto-detect which LLM provider to use based on API keys."""
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Prefer Anthropic if available
        if anthropic_key and anthropic_key.startswith("sk-ant-"):
            logger.info("Detected Anthropic API key")
            return "anthropic"
        elif openai_key and openai_key.startswith("sk-"):
            logger.info("Detected OpenAI API key")
            return "openai"
        else:
            logger.error("No valid API key found for Anthropic or OpenAI")
            raise ValueError(
                "Please set either ANTHROPIC_API_KEY or OPENAI_API_KEY in config/.env"
            )

    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                return anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                sys.exit(1)

        elif self.llm_provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                return openai.OpenAI(api_key=api_key)
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                sys.exit(1)

        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")

    def _build_prompt(self, trending_topics: List[str], topic_context: str = "") -> str:
        """
        Build the prompt for the LLM.

        Args:
            trending_topics: List of trending keywords/topics
            topic_context: Additional context about the topics

        Returns:
            The formatted prompt string
        """
        topics_str = ", ".join(trending_topics[:5])  # Use top 5 topics

        prompt = f"""You are writing a humorous dialogue script for two animated characters:

**Butcher** - A {self.butcher_personality}
**Nutsy** - A {self.nutsy_personality}

Create a short, funny conversation ({self.min_lines}-{self.max_lines} lines total) where they discuss these trending topics: {topics_str}

The tone should be: {self.content_tone}

Guidelines:
- Butcher should be sarcastic and skeptical, making witty observations
- Nutsy should be energetic and excitable, often misunderstanding things
- Keep the dialogue snappy and entertaining
- Make jokes about the trending topics
- Each line should be 1-2 sentences maximum
- Include emotional cues for voice acting

Output the script as a JSON array with this format:
[
  {{"character": "Butcher", "line": "dialogue text here", "emotion": "sarcastic"}},
  {{"character": "Nutsy", "line": "dialogue text here", "emotion": "excited"}},
  ...
]

Valid emotions: happy, sad, excited, sarcastic, angry, confused, surprised, neutral

{topic_context}

Generate the script now:"""

        return prompt

    def generate_with_anthropic(self, prompt: str) -> str:
        """
        Generate script using Anthropic Claude.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            The generated response
        """
        try:
            model = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1500"))

            logger.info(f"Generating script with Claude ({model})...")

            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            response = message.content[0].text
            logger.info("Script generated successfully")
            return response

        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            raise

    def generate_with_openai(self, prompt: str) -> str:
        """
        Generate script using OpenAI GPT.

        Args:
            prompt: The prompt to send to GPT

        Returns:
            The generated response
        """
        try:
            model = os.getenv("LLM_MODEL", "gpt-4")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.8"))
            max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1500"))

            logger.info(f"Generating script with GPT ({model})...")

            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a creative scriptwriter for comedy shorts."},
                    {"role": "user", "content": prompt}
                ]
            )

            result = response.choices[0].message.content
            logger.info("Script generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise

    def generate_script(
        self,
        trending_topics: List[str],
        topic_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        Generate a dialogue script based on trending topics.

        Args:
            trending_topics: List of trending keywords
            topic_context: Additional context about the topics

        Returns:
            List of dialogue line dictionaries
        """
        if not trending_topics:
            raise ValueError("No trending topics provided")

        # Validate trending topics
        if not isinstance(trending_topics, list):
            raise TypeError("trending_topics must be a list")

        if not all(isinstance(topic, str) and topic.strip() for topic in trending_topics):
            raise ValueError("All trending topics must be non-empty strings")

        if len(trending_topics) > 10:
            logger.warning(f"Too many topics ({len(trending_topics)}), using first 10")
            trending_topics = trending_topics[:10]

        prompt = self._build_prompt(trending_topics, topic_context)

        # Generate with appropriate provider
        if self.llm_provider == "anthropic":
            response = self.generate_with_anthropic(prompt)
        else:
            response = self.generate_with_openai(prompt)

        # Parse the JSON response
        script = self._parse_script(response)
        return script

    def _parse_script(self, response: str) -> List[Dict[str, str]]:
        """
        Parse the LLM response into a structured script.

        Args:
            response: Raw response from the LLM

        Returns:
            List of dialogue line dictionaries
        """
        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON array found in response")

            json_str = response[start_idx:end_idx]
            script = json.loads(json_str)

            # Validate the script format
            for i, line in enumerate(script):
                if not all(key in line for key in ['character', 'line', 'emotion']):
                    raise ValueError(f"Line {i} missing required fields")

                if line['character'] not in ['Butcher', 'Nutsy']:
                    raise ValueError(f"Invalid character name: {line['character']}")

            logger.info(f"Parsed script with {len(script)} lines")
            return script

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response was: {response}")
            raise ValueError("LLM response was not valid JSON")

    def save_script(self, script: List[Dict[str, str]], output_dir: str = "data/scripts") -> str:
        """
        Save the script to a JSON file.

        Args:
            script: The script to save
            output_dir: Directory to save the script in

        Returns:
            Path to the saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{timestamp}.json"
        filepath = output_path / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': timestamp,
                'characters': ['Butcher', 'Nutsy'],
                'script': script,
                'metadata': {
                    'llm_provider': self.llm_provider,
                    'line_count': len(script),
                    'generated_at': datetime.now().isoformat()
                }
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Script saved to {filepath}")
        return str(filepath)


def get_trending_topics(db_path: str, days: int = 7, limit: int = 5) -> List[str]:
    """
    Retrieve trending topics from the database.

    Args:
        db_path: Path to the trends database
        days: Number of days to look back
        limit: Maximum number of topics to retrieve

    Returns:
        List of trending keywords
    """
    db_file = Path(db_path)

    if not db_file.exists():
        logger.error(f"Trends database not found at {db_path}")
        logger.info("Run init_databases.py and fetch_trends.py first to populate the database")
        return []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT topic, relevance_score
            FROM trending_topics
            WHERE DATE(timestamp) >= DATE('now', '-' || ? || ' days')
            ORDER BY relevance_score DESC, timestamp DESC
            LIMIT ?
        """, (days, limit))

        rows = cursor.fetchall()
        topics = [row[0] for row in rows]

        if topics:
            logger.info(f"Retrieved {len(topics)} trending topics from database")
        else:
            logger.warning(f"No trending topics found in last {days} days")

        return topics


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate dialogue scripts for Butcher and Nutsy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate script from recent trends
  python generate_script.py

  # Generate script from specific topics
  python generate_script.py --topics "artificial intelligence" "climate change"

  # Use specific LLM provider
  python generate_script.py --provider anthropic

  # Show generated script without saving
  python generate_script.py --dry-run
        """
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to generate script about (overrides database trends)"
    )

    parser.add_argument(
        "--db-path",
        default="data/trends.db",
        help="Path to trends database (default: data/trends.db)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back for trends (default: 7)"
    )

    parser.add_argument(
        "--topic-limit",
        type=int,
        default=5,
        help="Maximum number of topics to include (default: 5)"
    )

    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        help="LLM provider to use (auto-detected if not specified)"
    )

    parser.add_argument(
        "--output-dir",
        default="data/scripts",
        help="Directory to save scripts (default: data/scripts)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and display script without saving"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Get topics
    if args.topics:
        topics = args.topics
        logger.info(f"Using provided topics: {', '.join(topics)}")
    else:
        topics = get_trending_topics(args.db_path, args.days, args.topic_limit)
        if not topics:
            logger.error("No topics available. Provide --topics or run fetch_trends.py first")
            sys.exit(1)

    # Generate script
    try:
        generator = ScriptGenerator(llm_provider=args.provider)
        script = generator.generate_script(topics)

        # Display the script
        print("\n" + "=" * 80)
        print("GENERATED SCRIPT")
        print("=" * 80)
        for i, line in enumerate(script, 1):
            print(f"\n{i}. {line['character']} [{line['emotion']}]:")
            print(f"   {line['line']}")
        print("\n" + "=" * 80)

        # Save unless dry-run
        if not args.dry_run:
            filepath = generator.save_script(script, args.output_dir)
            print(f"\nScript saved to: {filepath}")
        else:
            print("\n[DRY RUN] Script not saved")

    except Exception as e:
        logger.error(f"Failed to generate script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
