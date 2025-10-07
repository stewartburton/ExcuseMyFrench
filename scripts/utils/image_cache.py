#!/usr/bin/env python3
"""
Image cache manager for ComfyUI generated images.

This module provides caching functionality for ComfyUI-generated images,
using prompt hashing to avoid regenerating identical images.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from PIL import Image

logger = logging.getLogger(__name__)


class ImageCache:
    """Manages caching of ComfyUI-generated images."""

    def __init__(
        self,
        cache_dir: str = "data/comfyui_cache",
        cache_days: int = None,
        enabled: bool = None
    ):
        """
        Initialize the image cache.

        Args:
            cache_dir: Directory to store cached images
            cache_days: Number of days to keep cached images (None = use env var)
            enabled: Whether caching is enabled (None = use env var)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get settings from env vars if not specified
        if enabled is None:
            enabled = os.getenv("COMFYUI_CACHE_ENABLED", "true").lower() == "true"

        if cache_days is None:
            cache_days = int(os.getenv("COMFYUI_CACHE_DAYS", "30"))

        self.enabled = enabled
        self.cache_days = cache_days
        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        logger.info(f"Image cache initialized: {self.cache_dir}")
        logger.info(f"Cache enabled: {self.enabled}, retention: {self.cache_days} days")

    def _hash_params(self, params: Dict[str, Any]) -> str:
        """
        Create a hash from generation parameters.

        Args:
            params: Generation parameters dictionary

        Returns:
            SHA-256 hash string
        """
        # Sort keys for consistent hashing
        sorted_params = json.dumps(params, sort_keys=True)
        hash_obj = hashlib.sha256(sorted_params.encode('utf-8'))
        return hash_obj.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cached image."""
        return self.cache_dir / f"{cache_key}.png"

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get the file path for cache metadata."""
        return self.metadata_dir / f"{cache_key}.json"

    def get(self, params: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Retrieve an image from the cache.

        Args:
            params: Generation parameters to look up

        Returns:
            PIL Image if found and valid, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._hash_params(params)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        # Check if cached file exists
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key[:16]}...")
            return None

        # Check if metadata exists
        if not metadata_path.exists():
            logger.warning(f"Cache entry missing metadata: {cache_key[:16]}...")
            return None

        try:
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Check if cache entry has expired
            created_at = datetime.fromisoformat(metadata['created_at'])
            expiry_date = created_at + timedelta(days=self.cache_days)

            if datetime.now() > expiry_date:
                logger.info(f"Cache entry expired: {cache_key[:16]}...")
                self._remove_entry(cache_key)
                return None

            # Load and return image
            image = Image.open(cache_path)
            logger.info(f"Cache hit: {cache_key[:16]}...")

            # Update access time
            metadata['last_accessed'] = datetime.now().isoformat()
            metadata['access_count'] = metadata.get('access_count', 0) + 1

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return image

        except Exception as e:
            logger.error(f"Error loading cached image: {e}")
            self._remove_entry(cache_key)
            return None

    def put(
        self,
        params: Dict[str, Any],
        image: Image.Image,
        extra_metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store an image in the cache.

        Args:
            params: Generation parameters used
            image: PIL Image to cache
            extra_metadata: Additional metadata to store

        Returns:
            Cache key for the stored image
        """
        if not self.enabled:
            return ""

        cache_key = self._hash_params(params)
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        try:
            # Save image
            image.save(cache_path, format='PNG')

            # Create metadata
            metadata = {
                'cache_key': cache_key,
                'created_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0,
                'params': params,
                'image_size': image.size,
                'image_mode': image.mode
            }

            if extra_metadata:
                metadata.update(extra_metadata)

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Cached image: {cache_key[:16]}...")
            return cache_key

        except Exception as e:
            logger.error(f"Error caching image: {e}")
            return ""

    def _remove_entry(self, cache_key: str):
        """Remove a cache entry and its metadata."""
        cache_path = self._get_cache_path(cache_key)
        metadata_path = self._get_metadata_path(cache_key)

        if cache_path.exists():
            cache_path.unlink()

        if metadata_path.exists():
            metadata_path.unlink()

    def clear(self) -> int:
        """
        Clear all cached images.

        Returns:
            Number of entries removed
        """
        count = 0

        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.png"):
            cache_file.unlink()
            count += 1

        # Remove all metadata files
        for metadata_file in self.metadata_dir.glob("*.json"):
            metadata_file.unlink()

        logger.info(f"Cleared {count} cache entries")
        return count

    def cleanup_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        count = 0
        expiry_threshold = datetime.now() - timedelta(days=self.cache_days)

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                created_at = datetime.fromisoformat(metadata['created_at'])

                if created_at < expiry_threshold:
                    cache_key = metadata['cache_key']
                    self._remove_entry(cache_key)
                    count += 1
                    logger.debug(f"Removed expired entry: {cache_key[:16]}...")

            except Exception as e:
                logger.error(f"Error processing metadata file {metadata_file}: {e}")
                # Remove corrupted metadata
                metadata_file.unlink()

        logger.info(f"Cleaned up {count} expired cache entries")
        return count

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_entries = len(list(self.cache_dir.glob("*.png")))
        total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.png"))

        # Analyze metadata
        access_counts = []
        creation_dates = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                access_counts.append(metadata.get('access_count', 0))
                created_at = datetime.fromisoformat(metadata['created_at'])
                creation_dates.append(created_at)

            except Exception as e:
                logger.error(f"Error reading metadata: {e}")

        # Calculate stats
        stats = {
            'total_entries': total_entries,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'cache_dir': str(self.cache_dir),
            'enabled': self.enabled,
            'retention_days': self.cache_days
        }

        if access_counts:
            stats['avg_access_count'] = round(sum(access_counts) / len(access_counts), 2)
            stats['total_cache_hits'] = sum(access_counts)

        if creation_dates:
            oldest = min(creation_dates)
            newest = max(creation_dates)
            stats['oldest_entry'] = oldest.isoformat()
            stats['newest_entry'] = newest.isoformat()

        return stats

    def print_stats(self):
        """Print cache statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("COMFYUI IMAGE CACHE STATISTICS")
        print("=" * 60)
        print(f"Cache directory: {stats['cache_dir']}")
        print(f"Cache enabled: {stats['enabled']}")
        print(f"Retention period: {stats['retention_days']} days")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Total size: {stats['total_size_mb']} MB")

        if 'total_cache_hits' in stats:
            print(f"Total cache hits: {stats['total_cache_hits']}")
            print(f"Average access count: {stats['avg_access_count']}")

        if 'oldest_entry' in stats:
            print(f"Oldest entry: {stats['oldest_entry']}")
            print(f"Newest entry: {stats['newest_entry']}")

        print("=" * 60 + "\n")


def main():
    """Main entry point for cache management CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage ComfyUI image cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show cache statistics
  python image_cache.py --stats

  # Clean up expired entries
  python image_cache.py --cleanup

  # Clear entire cache
  python image_cache.py --clear

  # Set custom cache directory
  python image_cache.py --cache-dir /path/to/cache --stats
        """
    )

    parser.add_argument(
        "--cache-dir",
        default="data/comfyui_cache",
        help="Cache directory path (default: data/comfyui_cache)"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show cache statistics"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove expired cache entries"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear entire cache"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize cache
    cache = ImageCache(cache_dir=args.cache_dir)

    # Execute commands
    if args.stats:
        cache.print_stats()

    elif args.cleanup:
        count = cache.cleanup_expired()
        print(f"Removed {count} expired cache entries")

    elif args.clear:
        response = input("Are you sure you want to clear the entire cache? (yes/no): ")
        if response.lower() == 'yes':
            count = cache.clear()
            print(f"Cleared {count} cache entries")
        else:
            print("Operation cancelled")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
