#!/usr/bin/env python3
"""
Pipeline runner wrapper for n8n integration.

This script provides a simplified interface to run individual pipeline steps
or the complete pipeline from n8n workflows.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs pipeline steps with proper error handling and logging."""

    def __init__(self):
        """Initialize the pipeline runner."""
        self.scripts_dir = Path(__file__).parent.parent
        self.project_root = self.scripts_dir.parent
        self.data_dir = self.project_root / "data"

    def run_step(
        self,
        step_name: str,
        args: List[str] = None,
        timeout: int = 600
    ) -> Dict:
        """
        Run a single pipeline step.

        Args:
            step_name: Name of the step to run
            args: Additional arguments for the step
            timeout: Timeout in seconds

        Returns:
            Dictionary with execution results
        """
        args = args or []

        step_scripts = {
            "fetch_trends": "fetch_trends.py",
            "generate_script": "generate_script.py",
            "generate_audio": "generate_audio.py",
            "select_images": "select_images.py",
            "assemble_video": "assemble_video.py",
            "post_instagram": "post_instagram.py"
        }

        if step_name not in step_scripts:
            return {
                "success": False,
                "step": step_name,
                "error": f"Unknown step: {step_name}",
                "available_steps": list(step_scripts.keys())
            }

        script_path = self.scripts_dir / step_scripts[step_name]

        if not script_path.exists():
            return {
                "success": False,
                "step": step_name,
                "error": f"Script not found: {script_path}"
            }

        cmd = [sys.executable, str(script_path)] + args

        try:
            logger.info(f"Running step: {step_name}")
            logger.debug(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            return {
                "success": result.returncode == 0,
                "step": step_name,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timestamp": datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "step": step_name,
                "error": f"Step timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "step": step_name,
                "error": str(e)
            }

    def get_latest_file(self, pattern: str) -> Optional[str]:
        """
        Get the most recently created file matching a pattern.

        Args:
            pattern: Glob pattern to match files

        Returns:
            Path to latest file or None
        """
        files = list(self.data_dir.rglob(pattern))

        if not files:
            return None

        # Sort by modification time
        latest = max(files, key=lambda p: p.stat().st_mtime)
        return str(latest)

    def run_full_pipeline(
        self,
        skip_posting: bool = False,
        region: str = "united_states"
    ) -> Dict:
        """
        Run the complete pipeline from start to finish.

        Args:
            skip_posting: Skip Instagram posting step
            region: Region for trend fetching

        Returns:
            Dictionary with execution results for all steps
        """
        results = {
            "pipeline_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "started_at": datetime.now().isoformat(),
            "steps": {},
            "success": False
        }

        # Step 1: Fetch trends
        logger.info("=" * 80)
        logger.info("STEP 1: Fetching trending topics")
        logger.info("=" * 80)

        result = self.run_step("fetch_trends", ["--region", region])
        results["steps"]["fetch_trends"] = result

        if not result["success"]:
            logger.error("Failed to fetch trends")
            results["failed_at"] = "fetch_trends"
            return results

        # Step 2: Generate script
        logger.info("=" * 80)
        logger.info("STEP 2: Generating script")
        logger.info("=" * 80)

        result = self.run_step("generate_script", ["--days", "7", "--topic-limit", "5"])
        results["steps"]["generate_script"] = result

        if not result["success"]:
            logger.error("Failed to generate script")
            results["failed_at"] = "generate_script"
            return results

        # Find latest script
        script_file = self.get_latest_file("scripts/episode_*.json")
        if not script_file:
            logger.error("No script file found")
            results["failed_at"] = "generate_script"
            results["steps"]["generate_script"]["error"] = "Script file not found"
            return results

        logger.info(f"Using script: {script_file}")
        results["script_file"] = script_file

        # Step 3: Generate audio
        logger.info("=" * 80)
        logger.info("STEP 3: Generating audio")
        logger.info("=" * 80)

        result = self.run_step("generate_audio", [script_file], timeout=900)
        results["steps"]["generate_audio"] = result

        if not result["success"]:
            logger.error("Failed to generate audio")
            results["failed_at"] = "generate_audio"
            return results

        # Find latest timeline
        timeline_file = self.get_latest_file("audio/*/timeline.json")
        if not timeline_file:
            logger.error("No timeline file found")
            results["failed_at"] = "generate_audio"
            results["steps"]["generate_audio"]["error"] = "Timeline file not found"
            return results

        logger.info(f"Using timeline: {timeline_file}")
        results["timeline_file"] = timeline_file

        # Step 4: Select images
        logger.info("=" * 80)
        logger.info("STEP 4: Selecting images")
        logger.info("=" * 80)

        image_selections_file = str(self.data_dir / "image_selections_latest.json")
        result = self.run_step(
            "select_images",
            [script_file, "--output", image_selections_file]
        )
        results["steps"]["select_images"] = result

        if not result["success"]:
            logger.error("Failed to select images")
            results["failed_at"] = "select_images"
            return results

        results["image_selections_file"] = image_selections_file

        # Step 5: Assemble video
        logger.info("=" * 80)
        logger.info("STEP 5: Assembling video")
        logger.info("=" * 80)

        result = self.run_step(
            "assemble_video",
            ["--timeline", timeline_file, "--images", image_selections_file],
            timeout=1800  # 30 minutes
        )
        results["steps"]["assemble_video"] = result

        if not result["success"]:
            logger.error("Failed to assemble video")
            results["failed_at"] = "assemble_video"
            return results

        # Find latest video
        video_file = self.get_latest_file("final_videos/*.mp4")
        if not video_file:
            logger.error("No video file found")
            results["failed_at"] = "assemble_video"
            results["steps"]["assemble_video"]["error"] = "Video file not found"
            return results

        logger.info(f"Video created: {video_file}")
        results["video_file"] = video_file

        # Step 6: Post to Instagram (optional)
        if not skip_posting:
            logger.info("=" * 80)
            logger.info("STEP 6: Posting to Instagram")
            logger.info("=" * 80)

            result = self.run_step("post_instagram", [video_file], timeout=600)
            results["steps"]["post_instagram"] = result

            if not result["success"]:
                logger.warning("Failed to post to Instagram (continuing anyway)")
                results["post_instagram_skipped"] = False
                results["post_instagram_failed"] = True
            else:
                logger.info("Successfully posted to Instagram")
        else:
            logger.info("Skipping Instagram posting (--skip-posting flag)")
            results["post_instagram_skipped"] = True

        results["success"] = True
        results["completed_at"] = datetime.now().isoformat()

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run pipeline steps for n8n integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single step
  python pipeline_runner.py --step fetch_trends --args "--region united_states"

  # Run full pipeline
  python pipeline_runner.py --full

  # Run full pipeline without posting
  python pipeline_runner.py --full --skip-posting

  # Run with specific region
  python pipeline_runner.py --full --region india
        """
    )

    parser.add_argument(
        "--step",
        help="Name of step to run (fetch_trends, generate_script, etc.)"
    )

    parser.add_argument(
        "--args",
        help="Arguments to pass to the step (as a single quoted string)"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full pipeline"
    )

    parser.add_argument(
        "--skip-posting",
        action="store_true",
        help="Skip Instagram posting (for full pipeline)"
    )

    parser.add_argument(
        "--region",
        default="united_states",
        help="Region for trend fetching (default: united_states)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for individual steps (default: 600)"
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

    runner = PipelineRunner()

    # Run pipeline
    if args.full:
        results = runner.run_full_pipeline(
            skip_posting=args.skip_posting,
            region=args.region
        )
    elif args.step:
        step_args = args.args.split() if args.args else []
        results = runner.run_step(args.step, step_args, args.timeout)
    else:
        parser.print_help()
        sys.exit(1)

    # Output results
    print(json.dumps(results, indent=2))

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
