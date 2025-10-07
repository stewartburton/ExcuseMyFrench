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
import re
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

        # Rate limiting tracking (Instagram limit: ~200 API calls per hour)
        self.rate_limit_window = 3600  # 1 hour in seconds
        self.max_calls_per_window = 180  # Conservative limit
        self.api_calls = []  # List of timestamps

        # Token refresh settings
        self.token_expires_in = None  # Will be loaded from token_info.json
        self.token_expires_at = None

        # Data directories
        self.data_dir = Path("data") / "instagram"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.data_dir / "posted_history.json"
        self.queue_file = self.data_dir / "queue.json"
        self.analytics_file = self.data_dir / "analytics.json"
        self.token_info_file = self.data_dir / "token_info.json"
        self.rate_limit_file = self.data_dir / "rate_limit.json"

        # Initialize data files
        self._init_data_files()

        # Load token info and rate limit data
        self._load_token_info()
        self._load_rate_limit_data()

        # Validate configuration
        if not dry_run:
            self._validate_config()
            # Check and refresh token if needed
            self._check_and_refresh_token()

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

        if not self.token_info_file.exists():
            self._save_json(self.token_info_file, {
                "access_token": None,
                "token_type": "long_lived",
                "expires_at": None,
                "last_refreshed": None
            })

        if not self.rate_limit_file.exists():
            self._save_json(self.rate_limit_file, {
                "api_calls": [],
                "last_reset": datetime.now().isoformat()
            })

    def _save_json(self, filepath: Path, data: Dict):
        """Save data to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, filepath: Path) -> Dict:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_token_info(self):
        """Load token information from file."""
        if self.token_info_file.exists():
            token_data = self._load_json(self.token_info_file)
            if token_data.get('expires_at'):
                self.token_expires_at = datetime.fromisoformat(token_data['expires_at'])

    def _save_token_info(self):
        """Save token information to file."""
        token_data = {
            "access_token": self.access_token,
            "token_type": "long_lived",
            "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
            "last_refreshed": datetime.now().isoformat()
        }
        self._save_json(self.token_info_file, token_data)

    def _check_and_refresh_token(self):
        """
        Check if access token is expiring soon and refresh if needed.

        Instagram long-lived tokens last 60 days. We refresh when <7 days remain.
        """
        if not self.access_token:
            return

        # If we have expiry info, check if token needs refresh
        if self.token_expires_at:
            days_until_expiry = (self.token_expires_at - datetime.now()).days

            if days_until_expiry < 7:
                logger.warning(f"Token expires in {days_until_expiry} days, attempting refresh...")
                self._refresh_access_token()
            else:
                logger.info(f"Token valid for {days_until_expiry} more days")
        else:
            # No expiry info - try to get it or set default (60 days for long-lived tokens)
            logger.info("No token expiry info found, setting default expiration")
            self.token_expires_at = datetime.now() + timedelta(days=60)
            self._save_token_info()

    def _refresh_access_token(self):
        """
        Refresh the Instagram access token.

        Uses the Meta Graph API to exchange the current long-lived token
        for a new one with extended validity.
        """
        try:
            logger.info("Refreshing Instagram access token...")

            # Endpoint for refreshing long-lived tokens
            refresh_url = f"{self.base_url}/oauth/access_token"

            params = {
                "grant_type": "ig_refresh_token",
                "access_token": self.access_token
            }

            response = requests.get(refresh_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'access_token' in data:
                self.access_token = data['access_token']
                # Long-lived tokens are valid for 60 days
                expires_in = data.get('expires_in', 60 * 24 * 60 * 60)  # Default 60 days in seconds
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

                self._save_token_info()
                logger.info(f"Token refreshed successfully, valid until {self.token_expires_at}")

                # Update environment variable for current session
                os.environ["META_ACCESS_TOKEN"] = self.access_token
            else:
                logger.error("Token refresh response missing access_token")

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            logger.warning("Continuing with existing token, but it may expire soon")

    def _load_rate_limit_data(self):
        """Load rate limit tracking data."""
        if self.rate_limit_file.exists():
            data = self._load_json(self.rate_limit_file)
            # Load recent API calls (within the window)
            cutoff_time = time.time() - self.rate_limit_window
            self.api_calls = [
                ts for ts in data.get('api_calls', [])
                if ts > cutoff_time
            ]

    def _save_rate_limit_data(self):
        """Save rate limit tracking data."""
        # Clean old timestamps outside the window
        cutoff_time = time.time() - self.rate_limit_window
        self.api_calls = [ts for ts in self.api_calls if ts > cutoff_time]

        data = {
            "api_calls": self.api_calls,
            "last_reset": datetime.now().isoformat()
        }
        self._save_json(self.rate_limit_file, data)

    def _check_rate_limit(self):
        """
        Check if we're approaching rate limits and wait if necessary.

        Implements proactive rate limiting to avoid hitting Instagram's limits.
        """
        # Clean old timestamps
        cutoff_time = time.time() - self.rate_limit_window
        self.api_calls = [ts for ts in self.api_calls if ts > cutoff_time]

        current_call_count = len(self.api_calls)

        if current_call_count >= self.max_calls_per_window:
            # Calculate how long to wait until oldest call expires
            oldest_call = min(self.api_calls)
            wait_time = int(oldest_call + self.rate_limit_window - time.time()) + 1

            logger.warning(
                f"Rate limit reached ({current_call_count}/{self.max_calls_per_window} calls). "
                f"Waiting {wait_time}s..."
            )
            time.sleep(wait_time)

            # Clean again after waiting
            cutoff_time = time.time() - self.rate_limit_window
            self.api_calls = [ts for ts in self.api_calls if ts > cutoff_time]

        # Record this API call
        self.api_calls.append(time.time())
        self._save_rate_limit_data()

        # Log current rate limit status
        remaining = self.max_calls_per_window - len(self.api_calls)
        logger.debug(f"Rate limit: {remaining}/{self.max_calls_per_window} calls remaining in window")

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
        Make HTTP request with exponential backoff retry logic and rate limiting.

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
        # Check rate limits before making request
        self._check_rate_limit()

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

    def _sanitize_caption(self, caption: str) -> Tuple[str, List[str]]:
        """
        Sanitize and validate Instagram caption.

        Instagram requirements:
        - Max 2,200 characters
        - Max 30 hashtags
        - Proper Unicode encoding

        Args:
            caption: Raw caption text

        Returns:
            Tuple of (sanitized_caption, list of warnings)
        """
        warnings = []
        sanitized = caption

        # Ensure proper encoding
        try:
            sanitized = sanitized.encode('utf-8').decode('utf-8')
        except UnicodeError as e:
            warnings.append(f"Encoding issue fixed: {e}")
            # Remove problematic characters
            sanitized = sanitized.encode('utf-8', errors='ignore').decode('utf-8')

        # Count hashtags
        hashtags = re.findall(r'#\w+', sanitized)
        if len(hashtags) > 30:
            warnings.append(f"Too many hashtags ({len(hashtags)}), Instagram max is 30")
            # Keep only first 30 hashtags
            hashtag_positions = [(m.start(), m.end()) for m in re.finditer(r'#\w+', sanitized)]
            if len(hashtag_positions) > 30:
                # Remove hashtags beyond the 30th
                excess_start = hashtag_positions[30][0]
                sanitized = sanitized[:excess_start].rstrip()
                warnings.append("Removed excess hashtags")

        # Check character limit
        if len(sanitized) > 2200:
            warnings.append(f"Caption too long ({len(sanitized)} chars), truncating to 2200")
            # Truncate at word boundary
            sanitized = sanitized[:2197]
            last_space = sanitized.rfind(' ')
            if last_space > 2000:  # Only truncate at word if close to limit
                sanitized = sanitized[:last_space]
            sanitized += "..."

        # Remove or replace problematic characters
        # Instagram doesn't like some special characters in captions
        problematic_chars = ['\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07', '\x08']
        for char in problematic_chars:
            if char in sanitized:
                sanitized = sanitized.replace(char, '')
                warnings.append("Removed null/control characters")

        return sanitized, warnings

    def _validate_video_file(self, video_path: str) -> Tuple[bool, str]:
        """
        Validate video file meets Instagram Reels requirements.

        Instagram Reels requirements:
        - File size: Max 4GB (API limit is often 100MB)
        - Duration: 3-90 seconds (optimal: 15-60s)
        - Format: MP4 or MOV
        - Codec: H.264 video, AAC audio
        - Aspect ratio: 9:16 (vertical) preferred
        - Resolution: Min 500x888, recommended 1080x1920

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        video_file = Path(video_path)

        if not video_file.exists():
            return False, f"Video file not found: {video_path}"

        # Check file size (Instagram API limit is typically 100MB, max is 4GB)
        file_size_mb = video_file.stat().st_size / (1024 * 1024)
        if file_size_mb > 4096:  # 4GB
            return False, f"Video file too large: {file_size_mb:.2f}MB (max 4GB)"

        if file_size_mb > 100:
            logger.warning(
                f"Video file is large ({file_size_mb:.2f}MB). "
                "Instagram API may reject files over 100MB."
            )

        # Check file extension
        if video_file.suffix.lower() not in ['.mp4', '.mov']:
            return False, f"Invalid video format: {video_file.suffix} (use .mp4 or .mov)"

        # Try to check video properties using ffmpeg if available
        try:
            import subprocess
            import json as json_module

            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-count_frames',
                    '-show_entries', 'stream=width,height,codec_name,duration,nb_frames,r_frame_rate',
                    '-show_entries', 'format=duration',
                    '-of', 'json',
                    str(video_file)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                probe_data = json_module.loads(result.stdout)

                # Check duration
                duration = None
                if 'format' in probe_data and 'duration' in probe_data['format']:
                    duration = float(probe_data['format']['duration'])
                elif 'streams' in probe_data and probe_data['streams']:
                    stream_duration = probe_data['streams'][0].get('duration')
                    if stream_duration:
                        duration = float(stream_duration)

                if duration:
                    if duration < 3:
                        return False, f"Video too short: {duration:.1f}s (min 3s for Reels)"
                    if duration > 90:
                        logger.warning(
                            f"Video is {duration:.1f}s long. "
                            "Instagram Reels work best with 15-60s videos."
                        )

                # Check codec
                if 'streams' in probe_data and probe_data['streams']:
                    stream = probe_data['streams'][0]
                    codec = stream.get('codec_name', '').lower()

                    if codec and codec not in ['h264', 'hevc']:
                        logger.warning(
                            f"Video codec is {codec}. "
                            "Instagram recommends H.264. Video may need re-encoding."
                        )

                    # Check resolution
                    width = stream.get('width')
                    height = stream.get('height')

                    if width and height:
                        if width < 500 or height < 888:
                            logger.warning(
                                f"Low resolution: {width}x{height}. "
                                "Instagram recommends minimum 500x888."
                            )

                        # Check aspect ratio (9:16 is ideal for Reels)
                        aspect_ratio = width / height
                        ideal_ratio = 9 / 16
                        if abs(aspect_ratio - ideal_ratio) > 0.1:
                            logger.warning(
                                f"Aspect ratio {width}:{height} ({aspect_ratio:.2f}). "
                                "Ideal for Reels is 9:16 (0.56)."
                            )

        except (subprocess.TimeoutExpired, FileNotFoundError, json_module.JSONDecodeError):
            # ffprobe not available or failed - continue with basic validation
            logger.debug("Could not probe video details with ffprobe")

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

        # Sanitize caption
        caption, caption_warnings = self._sanitize_caption(caption)
        if caption_warnings:
            for warning in caption_warnings:
                logger.warning(f"Caption sanitization: {warning}")

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
