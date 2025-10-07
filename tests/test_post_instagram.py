"""
Tests for Instagram posting functionality.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Import the module to test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from post_instagram import InstagramPoster


class TestCaptionSanitization:
    """Test caption sanitization functionality."""

    def test_sanitize_caption_basic(self, mock_env_vars):
        """Test basic caption sanitization."""
        poster = InstagramPoster(dry_run=True)

        caption = "This is a test caption #test #python"
        sanitized, warnings = poster._sanitize_caption(caption)

        assert sanitized == caption
        assert len(warnings) == 0

    def test_sanitize_caption_too_long(self, mock_env_vars):
        """Test caption truncation when too long."""
        poster = InstagramPoster(dry_run=True)

        # Create a caption longer than 2200 characters
        long_caption = "A" * 2300

        sanitized, warnings = poster._sanitize_caption(long_caption)

        assert len(sanitized) <= 2200
        assert any("too long" in w.lower() for w in warnings)
        assert sanitized.endswith("...")

    def test_sanitize_caption_too_many_hashtags(self, mock_env_vars):
        """Test hashtag limiting."""
        poster = InstagramPoster(dry_run=True)

        # Create 35 hashtags (exceeds limit of 30)
        hashtags = " ".join([f"#tag{i}" for i in range(35)])
        caption = f"Test caption {hashtags}"

        sanitized, warnings = poster._sanitize_caption(caption)

        # Count hashtags in result
        import re
        result_hashtags = re.findall(r'#\w+', sanitized)
        assert len(result_hashtags) <= 30
        assert any("hashtag" in w.lower() for w in warnings)

    def test_sanitize_caption_encoding(self, mock_env_vars):
        """Test handling of encoding issues."""
        poster = InstagramPoster(dry_run=True)

        # Caption with control characters
        caption = "Test\x00caption\x01with\x02problems"

        sanitized, warnings = poster._sanitize_caption(caption)

        # Control characters should be removed
        assert '\x00' not in sanitized
        assert '\x01' not in sanitized
        assert '\x02' not in sanitized


class TestVideoValidation:
    """Test video file validation."""

    def test_validate_nonexistent_file(self, mock_env_vars):
        """Test validation of nonexistent file."""
        poster = InstagramPoster(dry_run=True)

        is_valid, error = poster._validate_video_file("/nonexistent/file.mp4")

        assert not is_valid
        assert "not found" in error.lower()

    def test_validate_wrong_extension(self, mock_env_vars, temp_dir):
        """Test validation of wrong file extension."""
        poster = InstagramPoster(dry_run=True)

        # Create a file with wrong extension
        test_file = temp_dir / "test.avi"
        test_file.write_bytes(b"fake video data")

        is_valid, error = poster._validate_video_file(str(test_file))

        assert not is_valid
        assert "format" in error.lower()

    def test_validate_file_too_large(self, mock_env_vars, temp_dir):
        """Test validation of oversized file."""
        poster = InstagramPoster(dry_run=True)

        # Create a large fake file (over 4GB is impossible in test, so we check 100MB warning)
        test_file = temp_dir / "test.mp4"
        # We can't actually create a 4GB file in tests, so this tests the logic
        # In real scenario, would need to mock file size

        with patch('pathlib.Path.stat') as mock_stat:
            mock_stat.return_value.st_size = 5 * 1024 * 1024 * 1024  # 5GB

            is_valid, error = poster._validate_video_file(str(test_file))

            assert not is_valid
            assert "too large" in error.lower()


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_tracking(self, mock_env_vars, temp_dir):
        """Test that API calls are tracked."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        initial_count = len(poster.api_calls)

        # Simulate an API call
        poster._check_rate_limit()

        assert len(poster.api_calls) == initial_count + 1

    def test_rate_limit_enforcement(self, mock_env_vars, temp_dir):
        """Test that rate limit is enforced."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir
        poster.max_calls_per_window = 5  # Low limit for testing

        # Fill up to the limit
        poster.api_calls = [time.time()] * 5

        # This should trigger a wait
        with patch('time.sleep') as mock_sleep:
            poster._check_rate_limit()
            # Should have waited
            assert mock_sleep.called

    def test_rate_limit_cleanup(self, mock_env_vars, temp_dir):
        """Test that old timestamps are cleaned up."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        # Add old timestamps (outside the window)
        old_time = time.time() - poster.rate_limit_window - 100
        poster.api_calls = [old_time, old_time, old_time]

        poster._check_rate_limit()

        # Old timestamps should be cleaned, only the new one remains
        assert len(poster.api_calls) == 1


class TestTokenRefresh:
    """Test token refresh functionality."""

    def test_token_expiry_check(self, mock_env_vars, temp_dir):
        """Test checking if token is expiring."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        # Set token to expire in 5 days
        poster.token_expires_at = datetime.now() + timedelta(days=5)

        with patch.object(poster, '_refresh_access_token') as mock_refresh:
            poster._check_and_refresh_token()
            # Should trigger refresh (expires in <7 days)
            assert mock_refresh.called

    def test_token_not_expiring(self, mock_env_vars, temp_dir):
        """Test that fresh tokens are not refreshed."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        # Set token to expire in 30 days
        poster.token_expires_at = datetime.now() + timedelta(days=30)

        with patch.object(poster, '_refresh_access_token') as mock_refresh:
            poster._check_and_refresh_token()
            # Should NOT trigger refresh (plenty of time left)
            assert not mock_refresh.called


class TestDuplicateDetection:
    """Test duplicate video detection."""

    def test_file_hash_calculation(self, mock_env_vars, temp_dir):
        """Test that file hashing works consistently."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        # Create a test file
        test_file = temp_dir / "test.mp4"
        test_file.write_bytes(b"test video content")

        hash1 = poster._get_file_hash(str(test_file))
        hash2 = poster._get_file_hash(str(test_file))

        # Same file should produce same hash
        assert hash1 == hash2

    def test_duplicate_detection(self, mock_env_vars, temp_dir):
        """Test that duplicates are detected."""
        poster = InstagramPoster(dry_run=True)
        poster.data_dir = temp_dir

        # Create a test file
        test_file = temp_dir / "test.mp4"
        test_file.write_bytes(b"test video content")

        # Add to history
        file_hash = poster._get_file_hash(str(test_file))
        history_data = {
            "posted_videos": [
                {
                    "file_hash": file_hash,
                    "posted_at": datetime.now().isoformat()
                }
            ]
        }

        poster.history_file = temp_dir / "history.json"
        with open(poster.history_file, 'w') as f:
            json.dump(history_data, f)

        # Should detect as duplicate
        assert poster._is_duplicate(str(test_file))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
