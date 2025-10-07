#!/usr/bin/env python3
"""
Example: Complete workflow from script generation to Instagram posting.

This script demonstrates how to use all the components together to:
1. Generate a script from trending topics
2. Generate audio for the dialogue
3. Select images for each line
4. Assemble the final video
5. Generate an Instagram caption
6. Post to Instagram

This is a reference implementation showing how all pieces fit together.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_script_generation(topics=None):
    """
    Step 1: Generate dialogue script.

    Args:
        topics: Optional list of topics to use

    Returns:
        Path to generated script file
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Generating Script")
    logger.info("=" * 80)

    from generate_script import ScriptGenerator, get_trending_topics

    # Get topics
    if not topics:
        topics = get_trending_topics("data/trends.db", days=7, limit=5)
        if not topics:
            logger.warning("No trending topics found, using defaults")
            topics = ["artificial intelligence", "social media", "technology"]

    logger.info(f"Using topics: {', '.join(topics)}")

    # Generate script
    generator = ScriptGenerator()
    script = generator.generate_script(topics)
    script_path = generator.save_script(script)

    logger.info(f"Script generated: {script_path}")
    logger.info(f"Script has {len(script)} lines")

    return script_path


def run_audio_generation(script_path):
    """
    Step 2: Generate audio for dialogue.

    Args:
        script_path: Path to script JSON file

    Returns:
        Path to timeline JSON file
    """
    logger.info("=" * 80)
    logger.info("STEP 2: Generating Audio")
    logger.info("=" * 80)

    from generate_audio import AudioGenerator

    generator = AudioGenerator()
    timeline_path = generator.generate_audio_from_script(script_path)

    logger.info(f"Audio generated: {timeline_path}")

    return timeline_path


def run_image_selection(timeline_path):
    """
    Step 3: Select images for each dialogue line.

    Args:
        timeline_path: Path to timeline JSON file

    Returns:
        Path to image selections JSON file
    """
    logger.info("=" * 80)
    logger.info("STEP 3: Selecting Images")
    logger.info("=" * 80)

    from select_images import ImageSelector

    selector = ImageSelector()
    selections = selector.select_images_from_timeline(timeline_path)
    selections_path = selector.save_selections(selections, timeline_path)

    logger.info(f"Images selected: {selections_path}")

    return selections_path


def run_video_assembly(timeline_path, selections_path, music_path=None):
    """
    Step 4: Assemble the final video.

    Args:
        timeline_path: Path to timeline JSON
        selections_path: Path to image selections JSON
        music_path: Optional path to background music

    Returns:
        Path to final video file
    """
    logger.info("=" * 80)
    logger.info("STEP 4: Assembling Video")
    logger.info("=" * 80)

    from assemble_video import VideoAssembler

    assembler = VideoAssembler()
    video_path = assembler.assemble_video(
        timeline_path=timeline_path,
        image_selections_path=selections_path,
        music_path=music_path,
        add_subtitles=True
    )

    logger.info(f"Video assembled: {video_path}")

    return video_path


def run_caption_generation(video_path):
    """
    Step 5: Generate Instagram caption.

    Args:
        video_path: Path to video file

    Returns:
        Generated caption text
    """
    logger.info("=" * 80)
    logger.info("STEP 5: Generating Caption")
    logger.info("=" * 80)

    from generate_caption import CaptionGenerator

    generator = CaptionGenerator()
    caption = generator.generate_from_video(video_path)

    logger.info("Caption generated:")
    logger.info("-" * 80)
    logger.info(caption)
    logger.info("-" * 80)

    return caption


def run_instagram_posting(video_path, caption, dry_run=False):
    """
    Step 6: Post to Instagram.

    Args:
        video_path: Path to video file
        caption: Caption text
        dry_run: If True, simulate posting without uploading

    Returns:
        Instagram media ID if successful
    """
    logger.info("=" * 80)
    logger.info("STEP 6: Posting to Instagram")
    logger.info("=" * 80)

    from post_instagram import InstagramPoster

    poster = InstagramPoster(dry_run=dry_run)
    media_id = poster.upload_video(video_path, caption=caption)

    if media_id:
        if dry_run:
            logger.info("[DRY RUN] Would post video to Instagram")
        else:
            logger.info(f"Posted to Instagram! Media ID: {media_id}")
    else:
        logger.error("Failed to post to Instagram")

    return media_id


def main():
    """Main workflow execution."""
    parser = argparse.ArgumentParser(
        description="Complete workflow: Generate content and post to Instagram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow with dry run (no actual posting)
  python full_workflow_example.py --dry-run

  # Full workflow with actual posting
  python full_workflow_example.py

  # Use custom topics
  python full_workflow_example.py --topics "AI" "robots" "future"

  # Skip posting step (generate video only)
  python full_workflow_example.py --no-post

  # Use existing script
  python full_workflow_example.py --script data/scripts/episode_123.json

  # Add background music
  python full_workflow_example.py --music data/music/background.mp3
        """
    )

    parser.add_argument(
        "--topics",
        nargs="+",
        help="Topics to generate script about"
    )

    parser.add_argument(
        "--script",
        help="Use existing script file (skips script generation)"
    )

    parser.add_argument(
        "--music",
        help="Path to background music file"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview posting without actually uploading"
    )

    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Skip Instagram posting step"
    )

    parser.add_argument(
        "--queue",
        action="store_true",
        help="Add to posting queue instead of posting immediately"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    start_time = datetime.now()

    try:
        # Print banner
        print("\n" + "=" * 80)
        print("EXCUSE MY FRENCH - COMPLETE WORKFLOW")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")

        # Step 1: Script Generation (or use existing)
        if args.script:
            script_path = args.script
            logger.info(f"Using existing script: {script_path}")
        else:
            script_path = run_script_generation(topics=args.topics)

        # Step 2: Audio Generation
        timeline_path = run_audio_generation(script_path)

        # Step 3: Image Selection
        selections_path = run_image_selection(timeline_path)

        # Step 4: Video Assembly
        video_path = run_video_assembly(
            timeline_path,
            selections_path,
            music_path=args.music
        )

        # Step 5: Caption Generation
        caption = run_caption_generation(video_path)

        # Step 6: Instagram Posting (if not skipped)
        if not args.no_post:
            if args.queue:
                from post_instagram import InstagramPoster
                poster = InstagramPoster(dry_run=args.dry_run)
                poster.add_to_queue(video_path, caption=caption)
                logger.info("Video added to posting queue")
            else:
                media_id = run_instagram_posting(
                    video_path,
                    caption,
                    dry_run=args.dry_run
                )
        else:
            logger.info("Skipping Instagram posting (--no-post)")

        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Video file: {video_path}")
        if not args.no_post:
            if args.dry_run:
                print("Status: DRY RUN (not actually posted)")
            elif args.queue:
                print("Status: Added to queue")
            else:
                print("Status: Posted to Instagram")
        print("=" * 80 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
