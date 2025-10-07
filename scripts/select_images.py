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

    def _validate_image_path(self, file_path: str) -> bool:
        """
        Validate that an image path is within the allowed data directory.

        Args:
            file_path: Path to validate

        Returns:
            True if path is valid, False otherwise
        """
        try:
            # Get the base data directory (resolve to absolute path)
            base_dir = Path("data").resolve()

            # Resolve the file path to its canonical absolute path
            resolved_path = Path(file_path).resolve()

            # Check if the resolved path is within the base directory
            # is_relative_to() checks if this path starts with the base_dir path
            if hasattr(resolved_path, 'is_relative_to'):
                # Python 3.9+
                return resolved_path.is_relative_to(base_dir)
            else:
                # Python 3.8 fallback
                try:
                    resolved_path.relative_to(base_dir)
                    return True
                except ValueError:
                    return False
        except Exception as e:
            logger.error(f"Error validating path {file_path}: {e}")
            return False

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
        # Validate path to prevent traversal attacks
        if not self._validate_image_path(file_path):
            logger.error(f"Invalid image path (must be within data/ directory): {file_path}")
            return False

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
                # Verify file still exists and is within valid directory
                file_path = row[0]

                # Validate path to prevent traversal attacks
                if not self._validate_image_path(file_path):
                    logger.error(f"Invalid image path in database (not within data/ directory): {file_path}")
                    # Remove from database for security
                    cursor.execute("DELETE FROM image_library WHERE file_path = ?", (file_path,))
                    conn.commit()
                    return None

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

    def _batch_query_images(
        self,
        requests: List[Tuple[str, str]]
    ) -> Dict[Tuple[str, str], Optional[str]]:
        """
        Query for multiple images in a single database operation.

        Args:
            requests: List of (character, emotion) tuples

        Returns:
            Dictionary mapping (character, emotion) to image path
        """
        # Build unique character-emotion pairs
        unique_pairs = list(set(requests))

        if not unique_pairs:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Build IN clause for batch query
            placeholders = ','.join(['(?, ?)'] * len(unique_pairs))
            query = f"""
                SELECT character, emotion, file_path, usage_count, quality_score
                FROM image_library
                WHERE (character, emotion) IN (VALUES {placeholders})
                ORDER BY usage_count ASC, quality_score DESC, RANDOM()
            """

            # Flatten the pairs for the query
            params = [item for pair in unique_pairs for item in pair]

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Group results by (character, emotion)
            results = {}
            for row in rows:
                character, emotion, file_path, usage_count, quality_score = row
                key = (character, emotion)

                # Only keep the first (best) result for each pair
                if key not in results:
                    # Validate path
                    if self._validate_image_path(file_path) and Path(file_path).exists():
                        results[key] = file_path
                    else:
                        # Invalid or missing file, remove from database
                        cursor.execute("DELETE FROM image_library WHERE file_path = ?", (file_path,))

            conn.commit()
            return results

    def process_script(
        self,
        script: List[Dict[str, str]],
        fallback_emotions: Dict[str, List[str]] = None,
        use_batch: bool = True
    ) -> Dict:
        """
        Process a script and select images for each line.

        Args:
            script: List of dialogue line dictionaries
            fallback_emotions: Dictionary mapping emotions to fallback emotions
            use_batch: Use batch database queries for better performance

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

        if use_batch:
            return self._process_script_batch(script, fallback_emotions)
        else:
            return self._process_script_sequential(script, fallback_emotions)

    def _process_script_sequential(
        self,
        script: List[Dict[str, str]],
        fallback_emotions: Dict[str, List[str]]
    ) -> Dict:
        """Process script with individual queries (original behavior)."""
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

    def _process_script_batch(
        self,
        script: List[Dict[str, str]],
        fallback_emotions: Dict[str, List[str]]
    ) -> Dict:
        """Process script with batch database queries for better performance."""
        results = {
            'selections': [],
            'missing': [],
            'stats': {
                'total_lines': len(script),
                'images_found': 0,
                'images_missing': 0
            }
        }

        # Build all query requests (including fallbacks)
        all_requests = []
        line_requests = []  # Track which requests go with which line

        for i, line in enumerate(script, 1):
            character = line['character']
            emotion = line.get('emotion', 'neutral')

            # Build list of emotions to try (primary + fallbacks)
            emotions_to_try = [emotion]
            fallbacks = fallback_emotions.get(emotion.lower(), [])
            emotions_to_try.extend(fallbacks)

            # Add neutral as last resort if not already included
            if 'neutral' not in emotions_to_try:
                emotions_to_try.append('neutral')

            # Create request tuples
            line_reqs = [(character, emo) for emo in emotions_to_try]
            all_requests.extend(line_reqs)
            line_requests.append((i, line, line_reqs))

        # Execute batch query
        logger.info(f"Batch querying {len(set(all_requests))} unique image combinations")
        image_map = self._batch_query_images(all_requests)

        # Process results for each line
        for i, line, reqs in line_requests:
            character = line['character']
            emotion = line.get('emotion', 'neutral')

            # Find first matching image from request list
            image_path = None
            for req in reqs:
                if req in image_map:
                    image_path = image_map[req]
                    if req[1] != emotion:
                        logger.info(f"Line {i}: Using fallback emotion '{req[1]}' for {character}/{emotion}")
                    break

            if image_path:
                # Record usage
                self._record_usage(image_path)

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
