#!/usr/bin/env python3
"""
Main orchestration script for ExcuseMyFrench video generation pipeline.

This script runs the complete end-to-end pipeline or individual steps with
comprehensive logging, error handling, and resume capabilities.
"""

import argparse
import json
import logging
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
log_file = os.getenv("LOG_FILE", "data/logs/app.log")
log_dir = Path(log_file).parent
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the complete video generation pipeline."""

    def __init__(self, project_root: Path = None):
        """
        Initialize the pipeline orchestrator.

        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "scripts"
        self.data_dir = self.project_root / "data"

        # Pipeline state
        self.pipeline_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_file = self.data_dir / "pipeline_state" / f"{self.pipeline_id}.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Execution state
        self.current_state = {
            "pipeline_id": self.pipeline_id,
            "started_at": datetime.now().isoformat(),
            "steps_completed": [],
            "current_step": None,
            "artifacts": {},
            "status": "running"
        }

        logger.info(f"Pipeline initialized with ID: {self.pipeline_id}")

    def save_state(self):
        """Save current pipeline state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.current_state, f, indent=2)

    def load_state(self, pipeline_id: str) -> bool:
        """
        Load pipeline state from file.

        Args:
            pipeline_id: ID of pipeline to resume

        Returns:
            True if state loaded successfully, False otherwise
        """
        state_file = self.data_dir / "pipeline_state" / f"{pipeline_id}.json"

        if not state_file.exists():
            logger.error(f"State file not found: {state_file}")
            return False

        with open(state_file, 'r') as f:
            self.current_state = json.load(f)

        self.pipeline_id = pipeline_id
        self.state_file = state_file
        logger.info(f"Loaded state for pipeline: {pipeline_id}")
        return True

    def run_script(
        self,
        script_name: str,
        args: List[str] = None,
        timeout: int = 600
    ) -> Dict:
        """
        Run a Python script and capture output.

        Args:
            script_name: Name of the script to run
            args: Arguments to pass to the script
            timeout: Timeout in seconds

        Returns:
            Dictionary with execution results
        """
        args = args or []
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_path}"
            }

        cmd = [sys.executable, str(script_path)] + args

        try:
            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": ' '.join(cmd)
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Script timed out after {timeout} seconds")
            return {
                "success": False,
                "error": f"Timeout after {timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_latest_file(self, pattern: str) -> Optional[Path]:
        """
        Get the most recently created file matching a pattern.

        Args:
            pattern: Glob pattern to match

        Returns:
            Path to latest file or None
        """
        files = list(self.data_dir.rglob(pattern))

        if not files:
            return None

        return max(files, key=lambda p: p.stat().st_mtime)

    def step_fetch_trends(self, region: str = "united_states") -> bool:
        """
        Step 1: Fetch trending topics.

        Args:
            region: Region to fetch trends for

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching Trending Topics")
        logger.info("=" * 80)

        self.current_state["current_step"] = "fetch_trends"
        self.save_state()

        result = self.run_script("fetch_trends.py", ["--region", region])

        if result["success"]:
            logger.info("Successfully fetched trends")
            self.current_state["steps_completed"].append("fetch_trends")
            self.save_state()
            return True
        else:
            logger.error(f"Failed to fetch trends: {result.get('error') or result.get('stderr')}")
            return False

    def step_generate_script(self, days: int = 7, limit: int = 5) -> bool:
        """
        Step 2: Generate script from trends.

        Args:
            days: Number of days to look back
            limit: Maximum number of topics

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 2: Generating Script")
        logger.info("=" * 80)

        self.current_state["current_step"] = "generate_script"
        self.save_state()

        result = self.run_script(
            "generate_script.py",
            ["--days", str(days), "--topic-limit", str(limit)]
        )

        if result["success"]:
            # Find latest script
            script_file = self.get_latest_file("scripts/episode_*.json")

            if script_file:
                logger.info(f"Script generated: {script_file}")
                self.current_state["steps_completed"].append("generate_script")
                self.current_state["artifacts"]["script_file"] = str(script_file)
                self.save_state()
                return True
            else:
                logger.error("Script file not found after generation")
                return False
        else:
            logger.error(f"Failed to generate script: {result.get('error') or result.get('stderr')}")
            return False

    def step_generate_audio(self) -> bool:
        """
        Step 3: Generate audio from script.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 3: Generating Audio")
        logger.info("=" * 80)

        script_file = self.current_state["artifacts"].get("script_file")

        if not script_file:
            logger.error("No script file found in state")
            return False

        self.current_state["current_step"] = "generate_audio"
        self.save_state()

        result = self.run_script(
            "generate_audio.py",
            [script_file],
            timeout=900  # 15 minutes for audio generation
        )

        if result["success"]:
            # Find latest timeline
            timeline_file = self.get_latest_file("audio/*/timeline.json")

            if timeline_file:
                logger.info(f"Audio generated: {timeline_file}")
                self.current_state["steps_completed"].append("generate_audio")
                self.current_state["artifacts"]["timeline_file"] = str(timeline_file)
                self.save_state()
                return True
            else:
                logger.error("Timeline file not found after audio generation")
                return False
        else:
            logger.error(f"Failed to generate audio: {result.get('error') or result.get('stderr')}")
            return False

    def step_select_images(self) -> bool:
        """
        Step 4: Select images for each dialogue line.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 4: Selecting Images")
        logger.info("=" * 80)

        script_file = self.current_state["artifacts"].get("script_file")

        if not script_file:
            logger.error("No script file found in state")
            return False

        self.current_state["current_step"] = "select_images"
        self.save_state()

        selections_file = str(self.data_dir / f"image_selections_{self.pipeline_id}.json")

        result = self.run_script(
            "select_images.py",
            [script_file, "--output", selections_file]
        )

        if result["success"]:
            logger.info(f"Images selected: {selections_file}")
            self.current_state["steps_completed"].append("select_images")
            self.current_state["artifacts"]["selections_file"] = selections_file
            self.save_state()
            return True
        else:
            logger.error(f"Failed to select images: {result.get('error') or result.get('stderr')}")
            return False

    def step_assemble_video(self, music_path: Optional[str] = None) -> bool:
        """
        Step 5: Assemble final video.

        Args:
            music_path: Optional path to background music

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 5: Assembling Video")
        logger.info("=" * 80)

        timeline_file = self.current_state["artifacts"].get("timeline_file")
        selections_file = self.current_state["artifacts"].get("selections_file")

        if not timeline_file or not selections_file:
            logger.error("Missing timeline or selections file in state")
            return False

        self.current_state["current_step"] = "assemble_video"
        self.save_state()

        args = ["--timeline", timeline_file, "--images", selections_file]

        if music_path:
            args.extend(["--music", music_path])

        result = self.run_script(
            "assemble_video.py",
            args,
            timeout=1800  # 30 minutes for video assembly
        )

        if result["success"]:
            # Find latest video
            video_file = self.get_latest_file("final_videos/*.mp4")

            if video_file:
                logger.info(f"Video assembled: {video_file}")
                self.current_state["steps_completed"].append("assemble_video")
                self.current_state["artifacts"]["video_file"] = str(video_file)
                self.save_state()
                return True
            else:
                logger.error("Video file not found after assembly")
                return False
        else:
            logger.error(f"Failed to assemble video: {result.get('error') or result.get('stderr')}")
            return False

    def step_post_instagram(self) -> bool:
        """
        Step 6: Post video to Instagram.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("STEP 6: Posting to Instagram")
        logger.info("=" * 80)

        video_file = self.current_state["artifacts"].get("video_file")

        if not video_file:
            logger.error("No video file found in state")
            return False

        # Check if posting is enabled
        if os.getenv("INSTAGRAM_POST_ENABLED", "true").lower() != "true":
            logger.info("Instagram posting disabled in configuration")
            self.current_state["steps_completed"].append("post_instagram_skipped")
            self.save_state()
            return True

        self.current_state["current_step"] = "post_instagram"
        self.save_state()

        result = self.run_script(
            "post_instagram.py",
            [video_file],
            timeout=600  # 10 minutes for posting
        )

        if result["success"]:
            logger.info("Successfully posted to Instagram")
            self.current_state["steps_completed"].append("post_instagram")
            self.save_state()
            return True
        else:
            logger.warning(f"Failed to post to Instagram: {result.get('error') or result.get('stderr')}")
            # Don't fail the whole pipeline if posting fails
            self.current_state["steps_completed"].append("post_instagram_failed")
            self.save_state()
            return True

    def run_full_pipeline(
        self,
        region: str = "united_states",
        skip_posting: bool = False,
        music_path: Optional[str] = None
    ) -> bool:
        """
        Run the complete pipeline end-to-end.

        Args:
            region: Region for trend fetching
            skip_posting: Skip Instagram posting
            music_path: Optional path to background music

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info(f"STARTING PIPELINE: {self.pipeline_id}")
        logger.info("=" * 80)

        steps = [
            ("fetch_trends", lambda: self.step_fetch_trends(region)),
            ("generate_script", lambda: self.step_generate_script()),
            ("generate_audio", lambda: self.step_generate_audio()),
            ("select_images", lambda: self.step_select_images()),
            ("assemble_video", lambda: self.step_assemble_video(music_path))
        ]

        if not skip_posting:
            steps.append(("post_instagram", lambda: self.step_post_instagram()))

        for step_name, step_func in steps:
            # Skip if already completed
            if step_name in self.current_state["steps_completed"]:
                logger.info(f"Skipping {step_name} (already completed)")
                continue

            success = step_func()

            if not success:
                logger.error(f"Pipeline failed at step: {step_name}")
                self.current_state["status"] = "failed"
                self.current_state["failed_at"] = step_name
                self.current_state["completed_at"] = datetime.now().isoformat()
                self.save_state()
                return False

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        self.current_state["status"] = "completed"
        self.current_state["completed_at"] = datetime.now().isoformat()
        self.save_state()

        # Print summary
        self.print_summary()

        return True

    def print_summary(self):
        """Print pipeline execution summary."""
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Pipeline ID: {self.pipeline_id}")
        print(f"Status: {self.current_state['status']}")
        print(f"Started: {self.current_state['started_at']}")
        print(f"Completed: {self.current_state.get('completed_at', 'N/A')}")
        print(f"\nSteps completed: {len(self.current_state['steps_completed'])}")
        for step in self.current_state["steps_completed"]:
            print(f"  - {step}")

        print(f"\nArtifacts generated:")
        for name, path in self.current_state["artifacts"].items():
            print(f"  - {name}: {path}")

        print("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ExcuseMyFrench video generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py

  # Run full pipeline without posting
  python run_pipeline.py --skip-posting

  # Run with specific region
  python run_pipeline.py --region india

  # Resume a failed pipeline
  python run_pipeline.py --resume 20240101_120000

  # Run individual step
  python run_pipeline.py --step fetch_trends

  # Add background music
  python run_pipeline.py --music data/music/background.mp3
        """
    )

    parser.add_argument(
        "--step",
        choices=["fetch_trends", "generate_script", "generate_audio",
                 "select_images", "assemble_video", "post_instagram"],
        help="Run only a specific step"
    )

    parser.add_argument(
        "--region",
        default="united_states",
        help="Region for trend fetching (default: united_states)"
    )

    parser.add_argument(
        "--skip-posting",
        action="store_true",
        help="Skip Instagram posting step"
    )

    parser.add_argument(
        "--music",
        help="Path to background music file"
    )

    parser.add_argument(
        "--resume",
        help="Resume a failed pipeline by ID"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()

    # Resume if requested
    if args.resume:
        if not orchestrator.load_state(args.resume):
            logger.error(f"Failed to load state for pipeline: {args.resume}")
            sys.exit(1)

    # Dry run mode
    if args.dry_run:
        print("DRY RUN MODE - No actions will be executed")
        print(f"Pipeline ID: {orchestrator.pipeline_id}")
        print(f"Region: {args.region}")
        print(f"Skip posting: {args.skip_posting}")
        if args.music:
            print(f"Music: {args.music}")
        sys.exit(0)

    # Run pipeline
    try:
        if args.step:
            # Run single step
            step_methods = {
                "fetch_trends": lambda: orchestrator.step_fetch_trends(args.region),
                "generate_script": orchestrator.step_generate_script,
                "generate_audio": orchestrator.step_generate_audio,
                "select_images": orchestrator.step_select_images,
                "assemble_video": lambda: orchestrator.step_assemble_video(args.music),
                "post_instagram": orchestrator.step_post_instagram
            }

            success = step_methods[args.step]()
        else:
            # Run full pipeline
            success = orchestrator.run_full_pipeline(
                region=args.region,
                skip_posting=args.skip_posting,
                music_path=args.music
            )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        orchestrator.current_state["status"] = "interrupted"
        orchestrator.current_state["interrupted_at"] = datetime.now().isoformat()
        orchestrator.save_state()
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        orchestrator.current_state["status"] = "error"
        orchestrator.current_state["error"] = str(e)
        orchestrator.current_state["failed_at"] = datetime.now().isoformat()
        orchestrator.save_state()
        sys.exit(1)


if __name__ == "__main__":
    main()
