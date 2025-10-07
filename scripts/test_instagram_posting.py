#!/usr/bin/env python3
"""
Test script for Instagram posting functionality.

This script tests the Instagram posting system without actually posting
to ensure everything is configured correctly.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from post_instagram import InstagramPoster
from generate_caption import CaptionGenerator


def test_configuration():
    """Test that configuration is properly set up."""
    print("\n" + "=" * 80)
    print("Testing Configuration")
    print("=" * 80)

    try:
        poster = InstagramPoster(dry_run=True)
        print("✓ Configuration loaded successfully")
        print(f"  - Access Token: {'*' * 20} (hidden)")
        print(f"  - User ID: {poster.user_id}")
        print(f"  - Post Enabled: {poster.post_enabled}")
        print(f"  - Default Hashtags: {poster.default_hashtags}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False


def test_data_files():
    """Test that data files are properly initialized."""
    print("\n" + "=" * 80)
    print("Testing Data Files")
    print("=" * 80)

    data_dir = Path("data") / "instagram"
    files = ["posted_history.json", "queue.json", "analytics.json"]

    all_ok = True
    for filename in files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✓ {filename} exists and is valid JSON")
            except json.JSONDecodeError:
                print(f"✗ {filename} exists but is invalid JSON")
                all_ok = False
        else:
            print(f"✗ {filename} not found")
            all_ok = False

    return all_ok


def test_caption_generator():
    """Test caption generation."""
    print("\n" + "=" * 80)
    print("Testing Caption Generator")
    print("=" * 80)

    try:
        generator = CaptionGenerator()
        print("✓ CaptionGenerator initialized")

        # Test with sample script
        sample_script = [
            {
                "character": "Butcher",
                "line": "Did you see the latest AI news? They're saying robots will take over the world.",
                "emotion": "sarcastic"
            },
            {
                "character": "Nutsy",
                "line": "OH NO! Should I start learning binary?!",
                "emotion": "excited"
            },
            {
                "character": "Butcher",
                "line": "Nutsy, you can barely count to ten.",
                "emotion": "sarcastic"
            }
        ]

        caption = generator.generate_from_script(sample_script)
        print("✓ Caption generated successfully")
        print("\nGenerated Caption:")
        print("-" * 80)
        print(caption)
        print("-" * 80)

        return True
    except Exception as e:
        print(f"✗ Caption generation error: {e}")
        return False


def test_video_validation():
    """Test video file validation."""
    print("\n" + "=" * 80)
    print("Testing Video Validation")
    print("=" * 80)

    poster = InstagramPoster(dry_run=True)

    # Test non-existent file
    is_valid, error_msg = poster._validate_video_file("nonexistent.mp4")
    if not is_valid and "not found" in error_msg:
        print("✓ Correctly detects non-existent files")
    else:
        print("✗ Failed to detect non-existent files")
        return False

    # Look for actual video files to test
    video_dir = Path("data") / "final_videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            test_video = video_files[0]
            is_valid, error_msg = poster._validate_video_file(str(test_video))
            if is_valid:
                print(f"✓ Validated existing video: {test_video.name}")
            else:
                print(f"✗ Failed to validate {test_video.name}: {error_msg}")
                return False
        else:
            print("  (No videos found to test validation)")
    else:
        print("  (No videos directory found)")

    return True


def test_duplicate_detection():
    """Test duplicate video detection."""
    print("\n" + "=" * 80)
    print("Testing Duplicate Detection")
    print("=" * 80)

    poster = InstagramPoster(dry_run=True)

    # Look for a video file to test with
    video_dir = Path("data") / "final_videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            test_video = video_files[0]

            # Should not be a duplicate initially
            is_dup = poster._is_duplicate(str(test_video))
            if not is_dup:
                print(f"✓ Correctly identifies new video: {test_video.name}")
            else:
                print(f"  Video already in history (may have been posted before)")

            # Get file hash
            file_hash = poster._get_file_hash(str(test_video))
            print(f"✓ Generated file hash: {file_hash[:16]}...")

            return True
        else:
            print("  (No videos found to test with)")
            return True
    else:
        print("  (No videos directory found)")
        return True


def test_dry_run_post():
    """Test dry run posting."""
    print("\n" + "=" * 80)
    print("Testing Dry Run Posting")
    print("=" * 80)

    poster = InstagramPoster(dry_run=True)

    # Look for a video to test with
    video_dir = Path("data") / "final_videos"
    if video_dir.exists():
        video_files = list(video_dir.glob("*.mp4"))
        if video_files:
            test_video = video_files[0]
            print(f"Testing with: {test_video.name}")

            # Test dry run post
            media_id = poster.upload_video(
                str(test_video),
                caption="Test caption #test"
            )

            if media_id == "dry_run_media_id":
                print("✓ Dry run posting works correctly")
                return True
            else:
                print(f"✗ Unexpected result: {media_id}")
                return False
        else:
            print("  (No videos found to test with)")
            print("  Tip: Run assemble_video.py to create a video first")
            return True
    else:
        print("  (No videos directory found)")
        return True


def test_status():
    """Test status retrieval."""
    print("\n" + "=" * 80)
    print("Testing Status Retrieval")
    print("=" * 80)

    try:
        poster = InstagramPoster(dry_run=True)
        status = poster.get_status()

        print("✓ Status retrieved successfully")
        print(f"  - Total posts: {status['total_posts']}")
        print(f"  - Successful: {status['successful_posts']}")
        print(f"  - Failed: {status['failed_posts']}")
        print(f"  - Pending in queue: {status['pending_in_queue']}")
        print(f"  - In history: {status['total_in_history']}")

        return True
    except Exception as e:
        print(f"✗ Status retrieval error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Instagram Posting System Test Suite")
    print("=" * 80)
    print("\nThis script will test the Instagram posting functionality")
    print("without actually posting anything to Instagram.\n")

    tests = [
        ("Configuration", test_configuration),
        ("Data Files", test_data_files),
        ("Caption Generator", test_caption_generator),
        ("Video Validation", test_video_validation),
        ("Duplicate Detection", test_duplicate_detection),
        ("Dry Run Posting", test_dry_run_post),
        ("Status Retrieval", test_status),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 80 + "\n")

    if passed == total:
        print("All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Configure your Meta access token in config/.env")
        print("2. Run: python scripts/post_instagram.py --status")
        print("3. Post a video: python scripts/post_instagram.py data/final_videos/your_video.mp4 --dry-run")
        return 0
    else:
        print("Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
