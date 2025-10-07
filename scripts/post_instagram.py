#!/usr/bin/env python3
"""
Post videos to Instagram using the Meta Graph API.

This script provides functionality to:
- Upload videos to Instagram as Reels
- Generate engaging captions with hashtags
- Track posting history to avoid duplicates
- Handle rate limits with exponential backoff
- Manage a posting queue
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib

import requests
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


class InstagramPoster:
    """Handles posting videos to Instagram using Meta Graph API."""

    def __init__(self, dry_run: bool = False):
        """
        Initialize the InstagramPoster.

        Args:
            dry_run: If True, simulate posting without actually uploading
        """
        self.dry_run = dry_run

        # Load configuration
        self.access_token = os.getenv("META_ACCESS_TOKEN")
        self.user_id = os.getenv("INSTAGRAM_USER_ID")
        self.post_enabled = os.getenv("INSTAGRAM_POST_ENABLED", "true").lower() == "true"
        self.default_hashtags = os.getenv(
            "INSTAGRAM_HASHTAGS",
            "#frenchbulldog #comedy #funny #reels #dogsofinstagram #fyp"
        )

        # API settings
        self.api_version = "v21.0"
        self.base_url = f"https://graph.facebook.com/{self.api_version}"

        # Rate limiting
        self.max_retries = int(os.getenv("API_MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("API_RETRY_DELAY", "2"))

        # Data directories
        self.data_dir = Path("data") / "instagram"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.data_dir / "posted_history.json"
        self.queue_file = self.data_dir / "queue.json"
        self.analytics_file = self.data_dir / "analytics.json"

        # Initialize data files
        self._init_data_files()

        # Validate configuration
        if not dry_run:
            self._validate_config()

        logger.info(f"InstagramPoster initialized (dry_run={dry_run})")

    def _validate_config(self):
        """Validate required configuration."""
        if not self.access_token:
            raise ValueError(
                "META_ACCESS_TOKEN not set. Please configure in config/.env"
            )

        if not self.user_id:
            raise ValueError(
                "INSTAGRAM_USER_ID not set. Please configure in config/.env"
            )

        if self.access_token == "your-meta-access-token-here":
            raise ValueError(
                "META_ACCESS_TOKEN appears to be a placeholder. "
                "Please set a valid access token in config/.env"
            )

        if not self.post_enabled:
            logger.warning("Instagram posting is disabled (INSTAGRAM_POST_ENABLED=false)")

    def _init_data_files(self):
        """Initialize data files if they don't exist."""
        if not self.history_file.exists():
            self._save_json(self.history_file, {"posted_videos": []})

        if not self.queue_file.exists():
            self._save_json(self.queue_file, {"queue": []})

        if not self.analytics_file.exists():
            self._save_json(self.analytics_file, {
                "total_posts": 0,
                "successful_posts": 0,
                "failed_posts": 0,
                "last_post_time": None,
                "posts": []
            })

    def _save_json(self, filepath: Path, data: Dict):
        """Save data to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, filepath: Path) -> Dict:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_file_hash(self, video_path: str) -> str:
        """
        Calculate SHA-256 hash of video file.

        Args:
            video_path: Path to video file

        Returns:
            Hexadecimal hash string
        """
        sha256_hash = hashlib.sha256()
        with open(video_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _is_duplicate(self, video_path: str) -> bool:
        """
        Check if video has already been posted.

        Args:
            video_path: Path to video file

        Returns:
            True if video has been posted before
        """
        file_hash = self._get_file_hash(video_path)
        history = self._load_json(self.history_file)

        for entry in history["posted_videos"]:
            if entry.get("file_hash") == file_hash:
                logger.info(f"Duplicate detected: {video_path} (posted on {entry['posted_at']})")
                return True

        return False

    def _add_to_history(
        self,
        video_path: str,
        media_id: str,
        caption: str,
        status: str
    ):
        """
        Add posted video to history.

        Args:
            video_path: Path to video file
            media_id: Instagram media ID
            caption: Video caption
            status: Post status (success/failed)
        """
        history = self._load_json(self.history_file)

        entry = {
            "video_path": str(video_path),
            "file_hash": self._get_file_hash(video_path),
            "media_id": media_id,
            "caption": caption[:100] + "..." if len(caption) > 100 else caption,
            "status": status,
            "posted_at": datetime.now().isoformat()
        }

        history["posted_videos"].append(entry)
        self._save_json(self.history_file, history)
        logger.info(f"Added to history: {video_path}")

    def _update_analytics(self, status: str, video_path: str, media_id: Optional[str] = None):
        """
        Update posting analytics.

        Args:
            status: Post status (success/failed)
            video_path: Path to video file
            media_id: Instagram media ID if successful
        """
        analytics = self._load_json(self.analytics_file)

        analytics["total_posts"] += 1
        if status == "success":
            analytics["successful_posts"] += 1
        else:
            analytics["failed_posts"] += 1

        analytics["last_post_time"] = datetime.now().isoformat()

        analytics["posts"].append({
            "video_path": str(video_path),
            "media_id": media_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })

        self._save_json(self.analytics_file, analytics)

    def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        retry_count: int = 0
    ) -> requests.Response:
        """
        Make HTTP request with exponential backoff retry logic.

        Args:
            method: HTTP method (GET, POST)
            url: Request URL
            data: Request data
            files: Files to upload
            retry_count: Current retry attempt

        Returns:
            Response object

        Raises:
            requests.RequestException: If request fails after retries
        """
        try:
            if method == "GET":
                response = requests.get(url, params=data, timeout=30)
            elif method == "POST":
                if files:
                    response = requests.post(url, data=data, files=files, timeout=300)
                else:
                    response = requests.post(url, data=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Check for rate limiting
            if response.status_code == 429:
                if retry_count < self.max_retries:
                    wait_time = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    return self._make_request(method, url, data, files, retry_count + 1)
                else:
                    raise requests.RequestException("Max retries exceeded due to rate limiting")

            # Check for other errors
            if response.status_code >= 400:
                logger.error(f"API error: {response.status_code} - {response.text}")

                # Retry on server errors
                if response.status_code >= 500 and retry_count < self.max_retries:
                    wait_time = self.retry_delay * (2 ** retry_count)
                    logger.warning(f"Server error. Waiting {wait_time}s before retry {retry_count + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    return self._make_request(method, url, data, files, retry_count + 1)

            response.raise_for_status()
            return response

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            if retry_count < self.max_retries:
                wait_time = self.retry_delay * (2 ** retry_count)
                logger.warning(f"Waiting {wait_time}s before retry {retry_count + 1}/{self.max_retries}")
                time.sleep(wait_time)
                return self._make_request(method, url, data, files, retry_count + 1)
            raise

    def _validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate video file meets Instagram requirements.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        video_file = Path(video_path)

        if not video_file.exists():
            return False, f"Video file not found: {video_path}"

        # Check file size (Instagram limit is 100MB for Reels)
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            return False, f"Video file too large: {file_size_mb:.2f}MB (max 100MB)"

        # Check file extension
        if video_file.suffix.lower() not in ['.mp4', '.mov']:
            return False, f"Invalid video format: {video_file.suffix} (use .mp4 or .mov)"

        return True, ""

    def upload_video(
        self,
        video_path: str,
        caption: Optional[str] = None,
        cover_url: Optional[str] = None,
        share_to_feed: bool = True
    ) -> Optional[str]:
        """
        Upload a video to Instagram as a Reel.

        Args:
            video_path: Path to video file
            caption: Video caption (auto-generated if None)
            cover_url: URL to cover image (optional)
            share_to_feed: Whether to share to main feed

        Returns:
            Instagram media ID if successful, None otherwise
        """
        logger.info(f"Uploading video: {video_path}")

        # Validate video
        is_valid, error_msg = self._validate_video_file(video_path)
        if not is_valid:
            logger.error(f"Video validation failed: {error_msg}")
            return None

        # Check for duplicates
        if self._is_duplicate(video_path):
            logger.warning("Video already posted. Skipping.")
            return None

        # Generate caption if not provided
        if caption is None:
            from generate_caption import CaptionGenerator
            caption_gen = CaptionGenerator()
            caption = caption_gen.generate_from_video(video_path)
            logger.info(f"Generated caption: {caption[:50]}...")

        # Add default hashtags
        if self.default_hashtags and self.default_hashtags not in caption:
            caption = f"{caption}\n\n{self.default_hashtags}"

        if self.dry_run:
            logger.info("[DRY RUN] Would upload video with caption:")
            logger.info(caption)
            return "dry_run_media_id"

        if not self.post_enabled:
            logger.warning("Posting is disabled. Set INSTAGRAM_POST_ENABLED=true to enable")
            return None

        try:
            # Step 1: Create container for the video
            logger.info("Step 1/2: Creating video container...")

            container_url = f"{self.base_url}/{self.user_id}/media"
            container_data = {
                "access_token": self.access_token,
                "media_type": "REELS",
                "video_url": video_path,  # Note: This should be a publicly accessible URL
                "caption": caption,
                "share_to_feed": share_to_feed
            }

            if cover_url:
                container_data["cover_url"] = cover_url

            container_response = self._make_request("POST", container_url, data=container_data)
            container_result = container_response.json()

            if "id" not in container_result:
                logger.error(f"Failed to create container: {container_result}")
                return None

            container_id = container_result["id"]
            logger.info(f"Container created: {container_id}")

            # Step 2: Check container status and publish
            logger.info("Step 2/2: Publishing video...")

            # Poll for container status
            status_url = f"{self.base_url}/{container_id}"
            max_checks = 60  # Maximum checks (5 minutes with 5s intervals)
            check_count = 0

            while check_count < max_checks:
                status_response = self._make_request("GET", status_url, data={
                    "access_token": self.access_token,
                    "fields": "status_code"
                })
                status_result = status_response.json()

                status_code = status_result.get("status_code")
                logger.info(f"Container status: {status_code}")

                if status_code == "FINISHED":
                    break
                elif status_code == "ERROR":
                    logger.error(f"Container processing error: {status_result}")
                    return None

                time.sleep(5)
                check_count += 1

            if check_count >= max_checks:
                logger.error("Timeout waiting for container to finish processing")
                return None

            # Publish the container
            publish_url = f"{self.base_url}/{self.user_id}/media_publish"
            publish_data = {
                "access_token": self.access_token,
                "creation_id": container_id
            }

            publish_response = self._make_request("POST", publish_url, data=publish_data)
            publish_result = publish_response.json()

            if "id" not in publish_result:
                logger.error(f"Failed to publish: {publish_result}")
                return None

            media_id = publish_result["id"]
            logger.info(f"Video published successfully! Media ID: {media_id}")

            # Update tracking
            self._add_to_history(video_path, media_id, caption, "success")
            self._update_analytics("success", video_path, media_id)

            return media_id

        except Exception as e:
            logger.error(f"Error uploading video: {e}")
            self._update_analytics("failed", video_path)
            return None

    def add_to_queue(
        self,
        video_path: str,
        caption: Optional[str] = None,
        scheduled_time: Optional[str] = None
    ):
        """
        Add video to posting queue.

        Args:
            video_path: Path to video file
            caption: Video caption
            scheduled_time: ISO format datetime string for scheduled posting
        """
        queue = self._load_json(self.queue_file)

        entry = {
            "video_path": str(video_path),
            "caption": caption,
            "scheduled_time": scheduled_time or datetime.now().isoformat(),
            "added_at": datetime.now().isoformat(),
            "status": "pending"
        }

        queue["queue"].append(entry)
        self._save_json(self.queue_file, queue)
        logger.info(f"Added to queue: {video_path}")

    def process_queue(self, max_posts: int = 1):
        """
        Process videos in the queue.

        Args:
            max_posts: Maximum number of videos to post
        """
        queue = self._load_json(self.queue_file)
        posted_count = 0
        updated_queue = []

        for entry in queue["queue"]:
            if entry["status"] != "pending":
                updated_queue.append(entry)
                continue

            if posted_count >= max_posts:
                updated_queue.append(entry)
                continue

            # Check if scheduled time has passed
            scheduled_time = datetime.fromisoformat(entry["scheduled_time"])
            if scheduled_time > datetime.now():
                updated_queue.append(entry)
                continue

            # Post the video
            logger.info(f"Processing queue entry: {entry['video_path']}")
            media_id = self.upload_video(entry["video_path"], entry["caption"])

            if media_id:
                entry["status"] = "posted"
                entry["media_id"] = media_id
                entry["posted_at"] = datetime.now().isoformat()
                posted_count += 1
            else:
                entry["status"] = "failed"

            updated_queue.append(entry)

        queue["queue"] = updated_queue
        self._save_json(self.queue_file, queue)
        logger.info(f"Queue processed: {posted_count} videos posted")

    def get_status(self) -> Dict:
        """
        Get current posting status and statistics.

        Returns:
            Dictionary with status information
        """
        analytics = self._load_json(self.analytics_file)
        queue = self._load_json(self.queue_file)
        history = self._load_json(self.history_file)

        pending_count = sum(1 for entry in queue["queue"] if entry["status"] == "pending")

        status = {
            "total_posts": analytics["total_posts"],
            "successful_posts": analytics["successful_posts"],
            "failed_posts": analytics["failed_posts"],
            "last_post_time": analytics["last_post_time"],
            "pending_in_queue": pending_count,
            "total_in_history": len(history["posted_videos"]),
            "post_enabled": self.post_enabled
        }

        return status


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Post videos to Instagram using Meta Graph API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Post a single video
  python post_instagram.py video.mp4

  # Post with custom caption
  python post_instagram.py video.mp4 --caption "Check out this hilarious moment!"

  # Dry run (preview without posting)
  python post_instagram.py video.mp4 --dry-run

  # Add to queue for later posting
  python post_instagram.py video.mp4 --queue

  # Process queue
  python post_instagram.py --process-queue

  # Check posting status
  python post_instagram.py --status
        """
    )

    parser.add_argument(
        "video",
        nargs="?",
        help="Path to video file to post"
    )

    parser.add_argument(
        "--caption",
        help="Custom caption for the video"
    )

    parser.add_argument(
        "--cover-url",
        help="URL to cover image for the video"
    )

    parser.add_argument(
        "--no-feed",
        action="store_true",
        help="Don't share to main feed (Reels only)"
    )

    parser.add_argument(
        "--queue",
        action="store_true",
        help="Add to queue instead of posting immediately"
    )

    parser.add_argument(
        "--process-queue",
        action="store_true",
        help="Process videos in the queue"
    )

    parser.add_argument(
        "--max-posts",
        type=int,
        default=1,
        help="Maximum number of videos to post from queue (default: 1)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show posting status and statistics"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without actually posting"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize poster
    poster = InstagramPoster(dry_run=args.dry_run)

    # Handle different operations
    if args.status:
        status = poster.get_status()
        print("\n" + "=" * 80)
        print("INSTAGRAM POSTING STATUS")
        print("=" * 80)
        print(f"Total posts: {status['total_posts']}")
        print(f"Successful: {status['successful_posts']}")
        print(f"Failed: {status['failed_posts']}")
        print(f"Last post: {status['last_post_time'] or 'Never'}")
        print(f"Pending in queue: {status['pending_in_queue']}")
        print(f"Total in history: {status['total_in_history']}")
        print(f"Posting enabled: {status['post_enabled']}")
        print("=" * 80 + "\n")
        return

    if args.process_queue:
        logger.info("Processing queue...")
        poster.process_queue(max_posts=args.max_posts)
        return

    if not args.video:
        parser.print_help()
        print("\nError: Please provide a video file or use --status or --process-queue")
        sys.exit(1)

    # Validate video path
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    # Add to queue or post immediately
    if args.queue:
        poster.add_to_queue(
            str(video_path),
            caption=args.caption
        )
        print(f"\nVideo added to queue: {video_path}")
    else:
        media_id = poster.upload_video(
            str(video_path),
            caption=args.caption,
            cover_url=args.cover_url,
            share_to_feed=not args.no_feed
        )

        if media_id:
            print("\n" + "=" * 80)
            print("VIDEO POSTED SUCCESSFULLY")
            print("=" * 80)
            print(f"Video: {video_path}")
            print(f"Media ID: {media_id}")
            print("=" * 80 + "\n")
        else:
            print("\n" + "=" * 80)
            print("VIDEO POSTING FAILED")
            print("=" * 80)
            print(f"Video: {video_path}")
            print("Check logs for details")
            print("=" * 80 + "\n")
            sys.exit(1)


if __name__ == "__main__":
    main()
