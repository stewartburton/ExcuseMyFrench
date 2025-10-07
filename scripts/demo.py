#!/usr/bin/env python3
"""
ExcuseMyFrench - Environment Validation Demo Script

This script validates that your ExcuseMyFrench environment is set up correctly
without requiring API keys or external services. It checks:
- Python version and dependencies
- GPU/CUDA availability
- Directory structure
- Database files
- Training data
- System tools (FFmpeg)

Usage:
    python scripts/demo.py
    # OR
    make demo

No API keys required!
"""

import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from typing import Tuple, List

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text: str):
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}{'=' * 70}{RESET}")
    print(f"{BOLD}{BLUE}{text}{RESET}")
    print(f"{BOLD}{BLUE}{'=' * 70}{RESET}\n")


def print_check(name: str, passed: bool, details: str = ""):
    """Print a check result with status indicator."""
    # Use ASCII-safe symbols for Windows compatibility
    check = "[PASS]" if passed else "[FAIL]"
    color = GREEN if passed else RED
    status = f"{color}{check}{RESET}"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")


def check_python_version() -> Tuple[bool, str]:
    """Check Python version is 3.10+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor >= 10:
        return True, f"Python {version_str}"
    return False, f"Python {version_str} (requires 3.10+)"


def check_cuda_available() -> Tuple[bool, str]:
    """Check if CUDA/GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"GPU: {gpu_name} (CUDA {cuda_version})"
        else:
            return False, "CUDA not available (GPU training will not work)"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, f"Error checking CUDA: {str(e)}"


def check_package_installed(package_name: str) -> Tuple[bool, str]:
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        return True, f"{package_name} installed"
    except ImportError:
        return False, f"{package_name} not installed"


def check_ffmpeg() -> Tuple[bool, str]:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version from first line
            version_line = result.stdout.split('\n')[0]
            return True, version_line
        return False, "FFmpeg command failed"
    except FileNotFoundError:
        return False, "FFmpeg not found in PATH"
    except Exception as e:
        return False, f"Error checking FFmpeg: {str(e)}"


def check_directory_exists(path: Path) -> Tuple[bool, str]:
    """Check if a directory exists."""
    if path.exists() and path.is_dir():
        return True, f"{path} exists"
    return False, f"{path} does not exist"


def check_file_exists(path: Path) -> Tuple[bool, str]:
    """Check if a file exists."""
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        return True, f"{path.name} ({size_mb:.1f} MB)"
    return False, f"{path.name} not found"


def check_database(db_path: Path) -> Tuple[bool, str]:
    """Check database connectivity and structure."""
    if not db_path.exists():
        return False, f"{db_path.name} does not exist"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get table count
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        table_count = cursor.fetchone()[0]

        # Get database size
        size_mb = db_path.stat().st_size / (1024 * 1024)

        conn.close()
        return True, f"{db_path.name} ({table_count} tables, {size_mb:.2f} MB)"
    except Exception as e:
        return False, f"{db_path.name} error: {str(e)}"


def check_disk_space(path: Path, min_gb: int = 10) -> Tuple[bool, str]:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024**3)

        if free_gb >= min_gb:
            return True, f"{free_gb:.1f} GB free"
        return False, f"Only {free_gb:.1f} GB free (need {min_gb}+ GB)"
    except Exception as e:
        return False, f"Error checking disk space: {str(e)}"


def count_training_images(training_dir: Path) -> Tuple[bool, str]:
    """Count training images in directory."""
    if not training_dir.exists():
        return False, f"{training_dir} does not exist"

    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_count = sum(
        1 for f in training_dir.glob('*')
        if f.suffix.lower() in image_extensions
    )

    if image_count >= 15:
        return True, f"{image_count} training images (good)"
    elif image_count > 0:
        return True, f"{image_count} training images (need 15+ for best results)"
    return False, "No training images found"


def main():
    """Run all validation checks."""
    # Get project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    print(f"\n{BOLD}ExcuseMyFrench Environment Validation{RESET}")
    print(f"Project root: {project_root}\n")

    # Track overall status
    all_critical_passed = True
    warnings = []

    # ========================================
    # System Checks
    # ========================================
    print_header("System Checks")

    passed, details = check_python_version()
    print_check("Python Version", passed, details)
    if not passed:
        all_critical_passed = False

    passed, details = check_cuda_available()
    print_check("GPU/CUDA Available", passed, details)
    if not passed:
        warnings.append("GPU training will not work without CUDA")

    passed, details = check_ffmpeg()
    print_check("FFmpeg Installed", passed, details)
    if not passed:
        all_critical_passed = False

    passed, details = check_disk_space(project_root, min_gb=10)
    print_check("Disk Space", passed, details)
    if not passed:
        warnings.append("Low disk space may cause issues")

    # ========================================
    # Python Packages
    # ========================================
    print_header("Required Python Packages")

    critical_packages = [
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
    ]

    for package in critical_packages:
        passed, details = check_package_installed(package)
        print_check(package, passed, details)
        if not passed:
            all_critical_passed = False

    optional_packages = [
        "elevenlabs",
        "anthropic",
        "openai",
    ]

    print(f"\n  {BOLD}Optional Packages (for full pipeline):{RESET}")
    for package in optional_packages:
        passed, details = check_package_installed(package)
        status = "[+]" if passed else "[ ]"
        print(f"    {status} {package}")

    # ========================================
    # Directory Structure
    # ========================================
    print_header("Directory Structure")

    required_dirs = [
        "data",
        "data/scripts",
        "data/audio",
        "data/images",
        "data/final_videos",
        "models",
        "training",
        "config",
    ]

    for dir_name in required_dirs:
        path = project_root / dir_name
        passed, details = check_directory_exists(path)
        print_check(dir_name, passed, details)
        if not passed and dir_name in ["data", "config"]:
            all_critical_passed = False

    # ========================================
    # Database Files
    # ========================================
    print_header("Database Files")

    databases = [
        "data/trends.db",
        "data/metrics.db",
        "data/image_library.db",
    ]

    for db_name in databases:
        path = project_root / db_name
        passed, details = check_database(path)
        print_check(db_name, passed, details)
        if not passed:
            warnings.append(f"Run 'python scripts/init_databases.py' to create {db_name}")

    # ========================================
    # Training Data
    # ========================================
    print_header("Training Data")

    training_dirs = [
        ("Butcher Images", "training/butcher/images"),
        ("Butcher Class Images", "training/butcher/class_images"),
    ]

    for name, dir_path in training_dirs:
        path = project_root / dir_path
        if "class_images" in dir_path:
            # Class images are generated during training
            if path.exists():
                passed, details = count_training_images(path)
                print_check(name, True, details)
            else:
                print_check(name, True, "Will be generated during training")
        else:
            passed, details = count_training_images(path)
            print_check(name, passed, details)
            if not passed:
                warnings.append(f"Add training images to {dir_path}")

    # ========================================
    # Configuration Files
    # ========================================
    print_header("Configuration Files")

    config_files = [
        "config/.env.example",
        "training/config/butcher_config.yaml",
    ]

    for config_name in config_files:
        path = project_root / config_name
        passed, details = check_file_exists(path)
        print_check(config_name, passed, details)

    # Check if .env exists (optional but recommended)
    env_file = project_root / "config/.env"
    if env_file.exists():
        print_check("config/.env", True, "Configuration file exists")
    else:
        print_check("config/.env", True, "Not created yet (copy from .env.example)")
        warnings.append("Copy config/.env.example to config/.env and add your API keys")

    # ========================================
    # Summary
    # ========================================
    print_header("Summary")

    if all_critical_passed:
        print(f"  {GREEN}[OK] All critical checks passed!{RESET}\n")
    else:
        print(f"  {RED}[!!] Some critical checks failed. Please fix the issues above.{RESET}\n")

    if warnings:
        print(f"  {YELLOW}[!] Warnings:{RESET}")
        for warning in warnings:
            print(f"    • {warning}")
        print()

    # ========================================
    # Next Steps
    # ========================================
    print_header("Next Steps")

    if not all_critical_passed:
        print("  1. Fix the failed critical checks above")
        print("  2. Install missing packages: pip install -r requirements.txt")
        print("  3. Create missing directories: make setup")
        print("  4. Re-run this demo: python scripts/demo.py")
    else:
        print("  Your environment is ready! Here's what you can do:")
        print()
        print("  [*] Basic Setup:")
        print("    1. Copy config/.env.example to config/.env")
        print("    2. Add your API keys to config/.env")
        print("    3. Run: make check-env")
        print()
        print("  [*] Train DreamBooth Model:")
        print("    1. Add 15-25 training images to training/butcher/images/")
        print("    2. Run: python scripts/train_dreambooth.py --config training/config/butcher_config.yaml")
        print("    3. Training takes ~1.5-2 hours on RTX 4070")
        print()
        print("  [*] Generate Videos:")
        print("    1. Configure API keys (OpenAI/Anthropic, ElevenLabs)")
        print("    2. Run: make run-pipeline")
        print("    3. Check output: data/final_videos/")
        print()
        print("  [*] Documentation:")
        print("    - Quick Start: QUICKSTART.md")
        print("    - Testing Guide: docs/TESTING.md")
        print("    - Troubleshooting: docs/TROUBLESHOOTING.md")

    print()

    # Return exit code
    sys.exit(0 if all_critical_passed else 1)


if __name__ == "__main__":
    main()
