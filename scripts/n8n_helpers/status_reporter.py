#!/usr/bin/env python3
"""
Status reporter for n8n monitoring integration.

This script performs various health checks and reports status back to n8n.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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


class StatusReporter:
    """Reports system status and health metrics."""

    def __init__(self):
        """Initialize the status reporter."""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"

    def check_database_health(self) -> Dict:
        """
        Check health of all databases.

        Returns:
            Dictionary with database health status
        """
        databases = {
            "trends": os.getenv("TRENDS_DB_PATH", "data/trends.db"),
            "metrics": os.getenv("METRICS_DB_PATH", "data/metrics.db"),
            "image_library": os.getenv("IMAGE_LIBRARY_DB_PATH", "data/image_library.db")
        }

        results = {
            "overall_status": "healthy",
            "databases": {}
        }

        for db_name, db_path in databases.items():
            db_file = self.project_root / db_path

            if not db_file.exists():
                results["databases"][db_name] = {
                    "status": "missing",
                    "path": str(db_path),
                    "exists": False
                }
                results["overall_status"] = "warning"
                continue

            try:
                # Try to connect and perform simple query
                with sqlite3.connect(db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
                    table_count = cursor.fetchone()[0]

                    # Get database size
                    db_size = db_file.stat().st_size

                    results["databases"][db_name] = {
                        "status": "healthy",
                        "path": str(db_path),
                        "exists": True,
                        "table_count": table_count,
                        "size_bytes": db_size,
                        "size_mb": round(db_size / (1024 * 1024), 2)
                    }

            except sqlite3.Error as e:
                results["databases"][db_name] = {
                    "status": "error",
                    "path": str(db_path),
                    "exists": True,
                    "error": str(e)
                }
                results["overall_status"] = "unhealthy"

        return results

    def check_api_quotas(self) -> Dict:
        """
        Check API quota usage and limits.

        Returns:
            Dictionary with API quota information
        """
        results = {
            "overall_status": "ok",
            "apis": {}
        }

        # ElevenLabs quota check
        elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        if elevenlabs_key:
            try:
                from elevenlabs import ElevenLabs
                client = ElevenLabs(api_key=elevenlabs_key)

                # Get subscription info
                try:
                    user = client.user.get()
                    subscription = user.subscription

                    character_count = subscription.character_count
                    character_limit = subscription.character_limit
                    usage_percent = (character_count / character_limit * 100) if character_limit > 0 else 0

                    results["apis"]["elevenlabs"] = {
                        "status": "ok" if usage_percent < 90 else "warning",
                        "character_count": character_count,
                        "character_limit": character_limit,
                        "usage_percent": round(usage_percent, 2),
                        "can_extend": subscription.can_extend_character_limit
                    }

                    if usage_percent >= 90:
                        results["overall_status"] = "warning"

                except Exception as e:
                    results["apis"]["elevenlabs"] = {
                        "status": "error",
                        "error": str(e)
                    }
                    results["overall_status"] = "warning"

            except ImportError:
                results["apis"]["elevenlabs"] = {
                    "status": "not_configured",
                    "message": "elevenlabs package not installed"
                }

        # OpenAI quota check (simplified - requires additional API calls)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.startswith("sk-"):
            results["apis"]["openai"] = {
                "status": "configured",
                "message": "API key configured (quota check requires additional implementation)"
            }

        # Anthropic quota check (simplified)
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.startswith("sk-ant-"):
            results["apis"]["anthropic"] = {
                "status": "configured",
                "message": "API key configured (quota check requires additional implementation)"
            }

        # Instagram API check
        meta_token = os.getenv("META_ACCESS_TOKEN")
        if meta_token:
            results["apis"]["instagram"] = {
                "status": "configured",
                "message": "Access token configured"
            }

        return results

    def check_storage_space(self) -> Dict:
        """
        Check available storage space.

        Returns:
            Dictionary with storage information
        """
        results = {
            "status": "ok",
            "directories": {}
        }

        # Check important directories
        dirs_to_check = [
            ("data", self.data_dir),
            ("scripts", self.project_root / "data" / "scripts"),
            ("audio", self.project_root / "data" / "audio"),
            ("final_videos", self.project_root / "data" / "final_videos"),
            ("images", self.project_root / "data" / "images")
        ]

        total_size = 0

        for dir_name, dir_path in dirs_to_check:
            if not dir_path.exists():
                results["directories"][dir_name] = {
                    "exists": False,
                    "path": str(dir_path)
                }
                continue

            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
            total_size += dir_size

            results["directories"][dir_name] = {
                "exists": True,
                "path": str(dir_path),
                "size_bytes": dir_size,
                "size_mb": round(dir_size / (1024 * 1024), 2),
                "size_gb": round(dir_size / (1024 * 1024 * 1024), 2)
            }

        results["total_size_bytes"] = total_size
        results["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        results["total_size_gb"] = round(total_size / (1024 * 1024 * 1024), 2)

        # Check if total size is concerning (>10GB warning, >50GB critical)
        if total_size > 50 * 1024 * 1024 * 1024:
            results["status"] = "critical"
        elif total_size > 10 * 1024 * 1024 * 1024:
            results["status"] = "warning"

        return results

    def get_recent_errors(self, log_file: Optional[str] = None, limit: int = 100) -> Dict:
        """
        Get recent errors from log files.

        Args:
            log_file: Path to log file (uses default if None)
            limit: Maximum number of log lines to check

        Returns:
            Dictionary with error information
        """
        if log_file is None:
            log_file = os.getenv("LOG_FILE", "data/logs/app.log")

        log_path = self.project_root / log_file

        results = {
            "status": "ok",
            "log_file": str(log_file),
            "errors": [],
            "warnings": [],
            "error_count": 0,
            "warning_count": 0
        }

        if not log_path.exists():
            results["status"] = "missing"
            results["message"] = f"Log file not found: {log_file}"
            return results

        try:
            # Read last N lines
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                recent_lines = lines[-limit:] if len(lines) > limit else lines

            # Parse errors and warnings
            for line in recent_lines:
                line_lower = line.lower()
                if ' error ' in line_lower or ' - error - ' in line_lower:
                    results["errors"].append(line.strip())
                    results["error_count"] += 1
                elif ' warning ' in line_lower or ' - warning - ' in line_lower:
                    results["warnings"].append(line.strip())
                    results["warning_count"] += 1

            # Set status based on error count
            if results["error_count"] > 20:
                results["status"] = "critical"
            elif results["error_count"] > 5:
                results["status"] = "warning"

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)

        return results

    def save_metrics(self, metrics_data: Dict):
        """
        Save monitoring metrics to database.

        Args:
            metrics_data: Metrics data to save
        """
        db_path = self.project_root / os.getenv("METRICS_DB_PATH", "data/metrics.db")

        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()

                # Create metrics table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS monitoring_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metric_type TEXT NOT NULL,
                        metric_data TEXT NOT NULL
                    )
                """)

                # Insert metrics
                cursor.execute("""
                    INSERT INTO monitoring_metrics (metric_type, metric_data)
                    VALUES (?, ?)
                """, ("system_health", json.dumps(metrics_data)))

                conn.commit()
                logger.info("Metrics saved to database")

        except sqlite3.Error as e:
            logger.error(f"Error saving metrics: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Report system status and health metrics"
    )

    parser.add_argument(
        "--check-database",
        action="store_true",
        help="Check database health"
    )

    parser.add_argument(
        "--check-api-quotas",
        action="store_true",
        help="Check API quota usage"
    )

    parser.add_argument(
        "--check-storage",
        action="store_true",
        help="Check storage space"
    )

    parser.add_argument(
        "--check-errors",
        action="store_true",
        help="Check recent errors in logs"
    )

    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to database"
    )

    parser.add_argument(
        "--data",
        help="JSON data to save (used with --save-metrics)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks"
    )

    parser.add_argument(
        "--output",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)"
    )

    args = parser.parse_args()

    reporter = StatusReporter()
    results = {}

    # Run checks
    if args.all or args.check_database:
        results["database"] = reporter.check_database_health()

    if args.all or args.check_api_quotas:
        results["api_quotas"] = reporter.check_api_quotas()

    if args.all or args.check_storage:
        results["storage"] = reporter.check_storage_space()

    if args.all or args.check_errors:
        results["errors"] = reporter.get_recent_errors()

    if args.save_metrics:
        if args.data:
            try:
                metrics_data = json.loads(args.data)
                reporter.save_metrics(metrics_data)
                results["metrics_saved"] = True
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON data: {e}")
                results["metrics_saved"] = False
        else:
            logger.error("--data required with --save-metrics")
            sys.exit(1)

    # Output results
    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        # Text output
        for check_name, check_results in results.items():
            print(f"\n{check_name.upper()}:")
            print(json.dumps(check_results, indent=2))

    # Exit with appropriate code
    if results:
        # Check for any unhealthy status
        has_error = False
        for check_results in results.values():
            if isinstance(check_results, dict):
                status = check_results.get("overall_status") or check_results.get("status")
                if status in ["unhealthy", "critical", "error"]:
                    has_error = True
                    break

        sys.exit(1 if has_error else 0)


if __name__ == "__main__":
    main()
