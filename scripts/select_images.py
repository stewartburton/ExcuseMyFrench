#!/usr/bin/env python3
"""
Select images from the image library database.

This script queries the image library database to find suitable images
for each character and emotion. It flags when images need to be generated.
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
import os

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageSelector:
    """Selects images from the library database."""

    def __init__(self, db_path: str = None):
        """
        Initialize the ImageSelector.

        Args:
            db_path: Path to the image library database
        """
        if db_path is None:
            db_path = os.getenv("IMAGE_LIBRARY_DB_PATH", "data/image_library.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Verify the database exists (schema should be created by init_databases.py)."""
        if not self.db_path.exists():
            logger.warning(
                f"Database not found at {self.db_path}. "
                "Run init_databases.py first to create the schema."
            )
        logger.info(f"Using database at {self.db_path}")

    def add_image(
        self,
        character: str,
        emotion: str,
        file_path: str,
        source: str = "manual",
        quality_score: float = 1.0,
        metadata: Dict = None
    ) -> bool:
        """
        Add an image to the library.

        Args:
            character: Character name (Butcher or Nutsy)
            emotion: Emotion tag
            file_path: Path to the image file
            source: Source of the image (manual, generated, etc.)
            quality_score: Quality rating (0.0 - 1.0)
            metadata: Additional metadata as dictionary

        Returns:
            True if added successfully, False otherwise
        """
        # Verify file exists
        if not Path(file_path).exists():
            logger.error(f"Image file not found: {file_path}")
            return False

        metadata_json = json.dumps(metadata) if metadata else None

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO image_library
                    (character, emotion, file_path, source, quality_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (character, emotion, file_path, source, quality_score, metadata_json))
                conn.commit()

                logger.info(f"Added image: {character}/{emotion} -> {file_path}")
                return True

        except sqlite3.IntegrityError:
            logger.warning(f"Image already exists in database: {file_path}")
            return False
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            return False

    def select_image(
        self,
        character: str,
        emotion: str,
        fallback_emotions: List[str] = None
    ) -> Optional[str]:
        """
        Select the best matching image for character and emotion.

        Args:
            character: Character name
            emotion: Desired emotion
            fallback_emotions: List of fallback emotions to try if exact match not found

        Returns:
            Path to selected image, or None if no suitable image found
        """
        # Try exact match first
        image_path = self._query_image(character, emotion)

        if image_path:
            self._record_usage(image_path)
            return image_path

        # Try fallback emotions
        if fallback_emotions:
            for fallback in fallback_emotions:
                image_path = self._query_image(character, fallback)
                if image_path:
                    logger.info(f"Using fallback emotion '{fallback}' for {character}/{emotion}")
                    self._record_usage(image_path)
                    return image_path

        # Try neutral as last resort
        if emotion != "neutral":
            image_path = self._query_image(character, "neutral")
            if image_path:
                logger.info(f"Using neutral emotion for {character}/{emotion}")
                self._record_usage(image_path)
                return image_path

        logger.warning(f"No suitable image found for {character}/{emotion}")
        return None

    def _query_image(self, character: str, emotion: str) -> Optional[str]:
        """
        Query for an image matching character and emotion.

        Args:
            character: Character name
            emotion: Emotion tag

        Returns:
            Path to image file or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Select least-used, highest-quality image
            cursor.execute("""
                SELECT file_path
                FROM image_library
                WHERE character = ? AND emotion = ?
                ORDER BY usage_count ASC, quality_score DESC, RANDOM()
                LIMIT 1
            """, (character, emotion))

            row = cursor.fetchone()

            if row:
                # Verify file still exists
                file_path = row[0]
                if Path(file_path).exists():
                    return file_path
                else:
                    logger.warning(f"Image file missing: {file_path}")
                    # Remove from database
                    cursor.execute("DELETE FROM image_library WHERE file_path = ?", (file_path,))
                    conn.commit()
                    return None
            else:
                return None

    def _record_usage(self, file_path: str):
        """Record that an image was used."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE image_library
                SET usage_count = usage_count + 1,
                    last_used_at = CURRENT_TIMESTAMP
                WHERE file_path = ?
            """, (file_path,))
            conn.commit()

    def process_script(
        self,
        script: List[Dict[str, str]],
        fallback_emotions: Dict[str, List[str]] = None
    ) -> Dict:
        """
        Process a script and select images for each line.

        Args:
            script: List of dialogue line dictionaries
            fallback_emotions: Dictionary mapping emotions to fallback emotions

        Returns:
            Dictionary with image selections and missing images list
        """
        if fallback_emotions is None:
            fallback_emotions = {
                "excited": ["happy", "surprised", "neutral"],
                "angry": ["sarcastic", "neutral"],
                "sad": ["neutral"],
                "sarcastic": ["neutral"],
                "confused": ["surprised", "neutral"],
                "surprised": ["excited", "happy", "neutral"],
                "happy": ["excited", "neutral"],
                "neutral": []
            }

        results = {
            'selections': [],
            'missing': [],
            'stats': {
                'total_lines': len(script),
                'images_found': 0,
                'images_missing': 0
            }
        }

        for i, line in enumerate(script, 1):
            character = line['character']
            emotion = line.get('emotion', 'neutral')

            fallbacks = fallback_emotions.get(emotion.lower(), ["neutral"])

            image_path = self.select_image(character, emotion, fallbacks)

            if image_path:
                results['selections'].append({
                    'line_index': i,
                    'character': character,
                    'emotion': emotion,
                    'image_path': image_path,
                    'found': True
                })
                results['stats']['images_found'] += 1
            else:
                results['selections'].append({
                    'line_index': i,
                    'character': character,
                    'emotion': emotion,
                    'image_path': None,
                    'found': False
                })
                results['missing'].append({
                    'character': character,
                    'emotion': emotion
                })
                results['stats']['images_missing'] += 1

                logger.warning(f"Line {i}: Missing image for {character}/{emotion}")

        return results

    def scan_directory(
        self,
        directory: str,
        character: str,
        emotion: str = "neutral",
        source: str = "manual"
    ) -> int:
        """
        Scan a directory and add all images to the library.

        Args:
            directory: Directory to scan
            character: Character these images belong to
            emotion: Default emotion tag for images
            source: Source tag for images

        Returns:
            Number of images added
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return 0

        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        added_count = 0

        for file_path in dir_path.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                if self.add_image(
                    character=character,
                    emotion=emotion,
                    file_path=str(file_path),
                    source=source
                ):
                    added_count += 1

        logger.info(f"Added {added_count} images from {directory}")
        return added_count

    def get_statistics(self) -> Dict:
        """
        Get statistics about the image library.

        Returns:
            Dictionary with library statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total images
            cursor.execute("SELECT COUNT(*) FROM image_library")
            total = cursor.fetchone()[0]

            # Images by character
            cursor.execute("""
                SELECT character, COUNT(*) as count
                FROM image_library
                GROUP BY character
            """)
            by_character = dict(cursor.fetchall())

            # Images by emotion
            cursor.execute("""
                SELECT emotion, COUNT(*) as count
                FROM image_library
                GROUP BY emotion
                ORDER BY count DESC
            """)
            by_emotion = dict(cursor.fetchall())

            # Coverage matrix
            cursor.execute("""
                SELECT character, emotion, COUNT(*) as count
                FROM image_library
                GROUP BY character, emotion
                ORDER BY character, emotion
            """)
            coverage = {}
            for char, emo, count in cursor.fetchall():
                if char not in coverage:
                    coverage[char] = {}
                coverage[char][emo] = count

            return {
                'total_images': total,
                'by_character': by_character,
                'by_emotion': by_emotion,
                'coverage': coverage
            }


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Select images from the image library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a script and select images
  python select_images.py data/scripts/episode_20240101.json

  # Scan a directory and add images
  python select_images.py --scan data/butcher_images --character Butcher --emotion neutral

  # Show library statistics
  python select_images.py --stats

  # Query for a specific image
  python select_images.py --query --character Butcher --emotion sarcastic
        """
    )

    parser.add_argument(
        "script_file",
        nargs="?",
        help="Path to script JSON file to process"
    )

    parser.add_argument(
        "--db-path",
        help="Path to image library database"
    )

    parser.add_argument(
        "--scan",
        help="Directory to scan and add images from"
    )

    parser.add_argument(
        "--character",
        choices=["Butcher", "Nutsy"],
        help="Character for scanned images or query"
    )

    parser.add_argument(
        "--emotion",
        default="neutral",
        help="Emotion tag for scanned images or query"
    )

    parser.add_argument(
        "--source",
        default="manual",
        help="Source tag for scanned images (default: manual)"
    )

    parser.add_argument(
        "--query",
        action="store_true",
        help="Query for a specific image"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display image library statistics"
    )

    parser.add_argument(
        "--output",
        help="Save results to JSON file"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize selector
    selector = ImageSelector(db_path=args.db_path)

    if args.stats:
        # Display statistics
        stats = selector.get_statistics()
        print("\n" + "=" * 80)
        print("IMAGE LIBRARY STATISTICS")
        print("=" * 80)
        print(f"Total images: {stats['total_images']}")
        print("\nBy character:")
        for char, count in stats['by_character'].items():
            print(f"  {char}: {count}")
        print("\nBy emotion:")
        for emotion, count in stats['by_emotion'].items():
            print(f"  {emotion}: {count}")
        print("\nCoverage matrix:")
        for char, emotions in stats['coverage'].items():
            print(f"  {char}:")
            for emotion, count in emotions.items():
                print(f"    {emotion}: {count}")
        print("=" * 80 + "\n")

    elif args.scan:
        # Scan directory
        if not args.character:
            logger.error("--character required when using --scan")
            sys.exit(1)

        count = selector.scan_directory(
            directory=args.scan,
            character=args.character,
            emotion=args.emotion,
            source=args.source
        )
        print(f"Added {count} images to library")

    elif args.query:
        # Query for specific image
        if not args.character:
            logger.error("--character required when using --query")
            sys.exit(1)

        image_path = selector.select_image(args.character, args.emotion)
        if image_path:
            print(f"Selected image: {image_path}")
        else:
            print(f"No image found for {args.character}/{args.emotion}")
            sys.exit(1)

    elif args.script_file:
        # Process script
        script_path = Path(args.script_file)
        if not script_path.exists():
            logger.error(f"Script file not found: {args.script_file}")
            sys.exit(1)

        with open(script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both direct script arrays and wrapped format
        if isinstance(data, list):
            script = data
        elif isinstance(data, dict) and 'script' in data:
            script = data['script']
        else:
            logger.error("Invalid script format")
            sys.exit(1)

        # Process script
        results = selector.process_script(script)

        # Display results
        print("\n" + "=" * 80)
        print("IMAGE SELECTION RESULTS")
        print("=" * 80)
        print(f"Total lines: {results['stats']['total_lines']}")
        print(f"Images found: {results['stats']['images_found']}")
        print(f"Images missing: {results['stats']['images_missing']}")

        if results['missing']:
            print("\nMissing images:")
            for item in results['missing']:
                print(f"  - {item['character']}/{item['emotion']}")

        print("=" * 80 + "\n")

        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
