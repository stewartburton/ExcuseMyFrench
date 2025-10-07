#!/usr/bin/env python3
"""
Webhook handlers for n8n integration.

This module provides Flask endpoints to receive webhook triggers from n8n
and forward them to the appropriate pipeline components.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Load environment variables
env_path = Path(__file__).parent.parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class PipelineWebhookHandler:
    """Handles webhook requests for pipeline operations."""

    def __init__(self):
        """Initialize the webhook handler."""
        self.scripts_dir = Path(__file__).parent.parent
        self.project_root = self.scripts_dir.parent

    def validate_api_key(self, request_key: str) -> bool:
        """
        Validate API key from webhook request.

        Args:
            request_key: API key from request

        Returns:
            True if valid, False otherwise
        """
        expected_key = os.getenv("N8N_API_KEY")
        if not expected_key:
            logger.warning("N8N_API_KEY not configured, allowing all requests")
            return True

        return request_key == expected_key

    def trigger_pipeline_step(
        self,
        step_name: str,
        params: Dict = None
    ) -> Dict:
        """
        Trigger a specific pipeline step.

        Args:
            step_name: Name of the step to run
            params: Additional parameters for the step

        Returns:
            Dictionary with execution results
        """
        params = params or {}

        # Map step names to scripts
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
                "error": f"Unknown step: {step_name}",
                "available_steps": list(step_scripts.keys())
            }

        script_path = self.scripts_dir / step_scripts[step_name]

        if not script_path.exists():
            return {
                "success": False,
                "error": f"Script not found: {script_path}"
            }

        # Build command
        cmd = [sys.executable, str(script_path)]

        # Add parameters based on step
        if step_name == "fetch_trends":
            region = params.get("region", "united_states")
            cmd.extend(["--region", region])

        elif step_name == "generate_script":
            days = params.get("days", 7)
            limit = params.get("topic_limit", 5)
            cmd.extend(["--days", str(days), "--topic-limit", str(limit)])

        elif step_name == "generate_audio":
            script_file = params.get("script_file")
            if script_file:
                cmd.append(script_file)
            else:
                return {"success": False, "error": "script_file parameter required"}

        elif step_name == "select_images":
            script_file = params.get("script_file")
            output = params.get("output", "data/image_selections_latest.json")
            if script_file:
                cmd.extend([script_file, "--output", output])
            else:
                return {"success": False, "error": "script_file parameter required"}

        elif step_name == "assemble_video":
            timeline = params.get("timeline")
            images = params.get("images")
            if timeline and images:
                cmd.extend(["--timeline", timeline, "--images", images])
            else:
                return {"success": False, "error": "timeline and images parameters required"}

        elif step_name == "post_instagram":
            video_file = params.get("video_file")
            if video_file:
                cmd.append(video_file)
            else:
                return {"success": False, "error": "video_file parameter required"}

        # Execute command
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "step": step_name,
                "timestamp": datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Step {step_name} timed out after 10 minutes",
                "step": step_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "step": step_name
            }


# Global handler instance
handler = PipelineWebhookHandler()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "excusemyfrench-webhook-handler",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/webhook/pipeline/<step_name>", methods=["POST"])
def handle_pipeline_step(step_name: str):
    """
    Handle webhook for a specific pipeline step.

    Args:
        step_name: Name of the pipeline step

    Returns:
        JSON response with execution results
    """
    # Validate API key
    api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not handler.validate_api_key(api_key):
        return jsonify({
            "success": False,
            "error": "Invalid API key"
        }), 401

    # Get parameters from request
    params = request.json or {}

    # Execute step
    result = handler.trigger_pipeline_step(step_name, params)

    status_code = 200 if result.get("success") else 500
    return jsonify(result), status_code


@app.route("/webhook/pipeline/full", methods=["POST"])
def handle_full_pipeline():
    """
    Handle webhook to run the full pipeline.

    Returns:
        JSON response with execution results
    """
    # Validate API key
    api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
    if not handler.validate_api_key(api_key):
        return jsonify({
            "success": False,
            "error": "Invalid API key"
        }), 401

    # Get parameters from request
    params = request.json or {}

    # Use run_pipeline.py for full pipeline
    script_path = handler.scripts_dir / "run_pipeline.py"

    if not script_path.exists():
        return jsonify({
            "success": False,
            "error": "run_pipeline.py not found"
        }), 500

    cmd = [sys.executable, str(script_path)]

    # Add optional parameters
    if params.get("skip_posting"):
        cmd.append("--skip-posting")

    try:
        logger.info(f"Executing full pipeline: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=handler.project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for full pipeline
        )

        return jsonify({
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp": datetime.now().isoformat()
        }), 200 if result.returncode == 0 else 500

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Full pipeline timed out after 1 hour"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run webhook handler server for n8n integration"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to bind to (default: 5000)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    args = parser.parse_args()

    logger.info(f"Starting webhook handler on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
