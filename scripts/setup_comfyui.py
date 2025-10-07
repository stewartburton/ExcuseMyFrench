#!/usr/bin/env python3
"""
Setup and configure ComfyUI with Wan 2.2 models for ExcuseMyFrench.

This script automates the installation and configuration of ComfyUI,
downloads the Wan 2.2 models, and sets up the proper directory structure.
"""

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


class ComfyUISetup:
    """Handles ComfyUI installation and configuration."""

    # ComfyUI GitHub repository
    COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"

    # Wan 2.2 model URLs (placeholder - update with actual URLs when available)
    WAN_MODELS = {
        "wan_2.2_base": {
            "url": "https://huggingface.co/Wan-AI/Wan-2.2-base/resolve/main/wan_2.2_base.safetensors",
            "size": "6.5GB",
            "sha256": None,  # Add checksum when known
            "required": True
        },
        "wan_2.2_anime": {
            "url": "https://huggingface.co/Wan-AI/Wan-2.2-anime/resolve/main/wan_2.2_anime.safetensors",
            "size": "6.5GB",
            "sha256": None,
            "required": False
        }
    }

    # Required custom nodes
    CUSTOM_NODES = {
        "comfyui-manager": {
            "repo": "https://github.com/ltdrdata/ComfyUI-Manager.git",
            "required": True
        },
        "comfyui-controlnet-aux": {
            "repo": "https://github.com/Fannovel16/comfyui_controlnet_aux.git",
            "required": False
        }
    }

    def __init__(self, install_path: Optional[str] = None, force: bool = False):
        """
        Initialize ComfyUI setup.

        Args:
            install_path: Custom installation path (uses env var if None)
            force: Force reinstallation even if already installed
        """
        self.force = force

        # Determine installation path
        if install_path:
            self.comfyui_path = Path(install_path)
        else:
            default_path = os.getenv("COMFYUI_PATH", "D:/ComfyUI")
            self.comfyui_path = Path(default_path)

        # Model paths
        self.models_path = self.comfyui_path / "models"
        self.wan_model_path = Path(os.getenv("WAN_MODEL_PATH", "models/wan2.2"))

        # Custom nodes path
        self.custom_nodes_path = self.comfyui_path / "custom_nodes"

        # Output path
        self.output_path = Path(os.getenv(
            "COMFYUI_OUTPUT_DIR",
            "data/comfyui_output"
        ))

        # Server settings
        self.server_url = os.getenv("COMFYUI_SERVER_URL", "http://127.0.0.1:8188")

        logger.info(f"Installation path: {self.comfyui_path}")
        logger.info(f"Server URL: {self.server_url}")

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        Check system prerequisites.

        Returns:
            Tuple of (success, list of missing items)
        """
        missing = []

        # Check Python version
        py_version = sys.version_info
        if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 10):
            missing.append(f"Python 3.10+ required (found {py_version.major}.{py_version.minor})")

        # Check git
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append("Git is not installed or not in PATH")

        # Check for CUDA (optional but recommended)
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                logger.warning("CUDA not available - will use CPU (much slower)")
        except ImportError:
            logger.warning("PyTorch not installed yet - will be installed with ComfyUI")

        # Check disk space (need ~20GB for ComfyUI + models)
        try:
            stat = shutil.disk_usage(self.comfyui_path.parent if self.comfyui_path.parent.exists() else ".")
            free_gb = stat.free / (1024**3)
            if free_gb < 20:
                missing.append(f"Insufficient disk space (need 20GB, have {free_gb:.1f}GB)")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        return len(missing) == 0, missing

    def clone_comfyui(self) -> bool:
        """
        Clone ComfyUI repository.

        Returns:
            True if successful, False otherwise
        """
        if self.comfyui_path.exists():
            if self.force:
                logger.info(f"Removing existing installation at {self.comfyui_path}")
                shutil.rmtree(self.comfyui_path)
            else:
                logger.info("ComfyUI already exists, skipping clone (use --force to reinstall)")
                return True

        try:
            logger.info(f"Cloning ComfyUI to {self.comfyui_path}")
            subprocess.run(
                ["git", "clone", self.COMFYUI_REPO, str(self.comfyui_path)],
                check=True,
                capture_output=True
            )
            logger.info("ComfyUI cloned successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone ComfyUI: {e.stderr.decode()}")
            return False

    def install_dependencies(self) -> bool:
        """
        Install ComfyUI Python dependencies.

        Returns:
            True if successful, False otherwise
        """
        requirements_file = self.comfyui_path / "requirements.txt"

        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False

        try:
            logger.info("Installing ComfyUI dependencies...")
            logger.info("This may take several minutes...")

            # Install PyTorch first with CUDA support if available
            if platform.system() == "Windows":
                torch_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ]
            else:
                torch_cmd = [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio"
                ]

            subprocess.run(torch_cmd, check=True)

            # Install other requirements
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                check=True
            )

            logger.info("Dependencies installed successfully")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False

    def setup_custom_nodes(self) -> bool:
        """
        Install custom nodes.

        Returns:
            True if successful, False otherwise
        """
        self.custom_nodes_path.mkdir(parents=True, exist_ok=True)

        for node_name, node_info in self.CUSTOM_NODES.items():
            node_path = self.custom_nodes_path / node_name

            if node_path.exists():
                logger.info(f"Custom node {node_name} already exists, skipping")
                continue

            try:
                logger.info(f"Installing custom node: {node_name}")
                subprocess.run(
                    ["git", "clone", node_info["repo"], str(node_path)],
                    check=True,
                    capture_output=True
                )

                # Install node-specific requirements if they exist
                node_requirements = node_path / "requirements.txt"
                if node_requirements.exists():
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(node_requirements)],
                        check=True,
                        capture_output=True
                    )

                logger.info(f"Installed {node_name}")

            except subprocess.CalledProcessError as e:
                if node_info.get("required", False):
                    logger.error(f"Failed to install required node {node_name}: {e}")
                    return False
                else:
                    logger.warning(f"Failed to install optional node {node_name}: {e}")

        return True

    def download_file(self, url: str, output_path: Path, expected_size: Optional[str] = None) -> bool:
        """
        Download a file with progress indicator.

        Args:
            url: URL to download from
            output_path: Where to save the file
            expected_size: Expected file size (for display only)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading {output_path.name}")
            if expected_size:
                logger.info(f"Expected size: {expected_size}")

            # Create progress callback
            def reporthook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    downloaded_mb = downloaded / (1024**2)
                    total_mb = total_size / (1024**2)
                    print(f"\rProgress: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end="")

            # Download file
            urllib.request.urlretrieve(url, output_path, reporthook)
            print()  # New line after progress
            logger.info(f"Downloaded {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def verify_checksum(self, file_path: Path, expected_sha256: str) -> bool:
        """
        Verify file checksum.

        Args:
            file_path: Path to file to verify
            expected_sha256: Expected SHA256 hash

        Returns:
            True if checksum matches, False otherwise
        """
        if not expected_sha256:
            logger.warning(f"No checksum available for {file_path.name}, skipping verification")
            return True

        try:
            logger.info(f"Verifying checksum for {file_path.name}")
            sha256_hash = hashlib.sha256()

            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            actual_hash = sha256_hash.hexdigest()

            if actual_hash == expected_sha256:
                logger.info("Checksum verified")
                return True
            else:
                logger.error(f"Checksum mismatch! Expected {expected_sha256}, got {actual_hash}")
                return False

        except Exception as e:
            logger.error(f"Failed to verify checksum: {e}")
            return False

    def download_wan_models(self) -> bool:
        """
        Download Wan 2.2 models.

        Returns:
            True if successful, False otherwise
        """
        # Create models directories
        checkpoints_path = self.models_path / "checkpoints"
        checkpoints_path.mkdir(parents=True, exist_ok=True)

        # Also create in our project structure
        self.wan_model_path.mkdir(parents=True, exist_ok=True)

        success = True
        for model_name, model_info in self.WAN_MODELS.items():
            model_file = checkpoints_path / f"{model_name}.safetensors"
            project_model = self.wan_model_path / f"{model_name}.safetensors"

            if model_file.exists():
                logger.info(f"Model {model_name} already exists, skipping")

                # Create symlink in project structure if it doesn't exist
                if not project_model.exists():
                    try:
                        if platform.system() == "Windows":
                            # On Windows, copy instead of symlink (requires admin for symlinks)
                            shutil.copy2(model_file, project_model)
                        else:
                            project_model.symlink_to(model_file)
                    except Exception as e:
                        logger.warning(f"Could not link model to project structure: {e}")

                continue

            # Download model
            if not self.download_file(model_info["url"], model_file, model_info.get("size")):
                if model_info.get("required", False):
                    logger.error(f"Failed to download required model {model_name}")
                    success = False
                else:
                    logger.warning(f"Failed to download optional model {model_name}")
                continue

            # Verify checksum if available
            if model_info.get("sha256"):
                if not self.verify_checksum(model_file, model_info["sha256"]):
                    logger.error(f"Checksum verification failed for {model_name}")
                    model_file.unlink()  # Delete corrupted file
                    if model_info.get("required", False):
                        success = False
                    continue

            # Copy to project structure
            try:
                if platform.system() == "Windows":
                    shutil.copy2(model_file, project_model)
                else:
                    project_model.symlink_to(model_file)
                logger.info(f"Linked model to {project_model}")
            except Exception as e:
                logger.warning(f"Could not link model to project structure: {e}")

        return success

    def create_config_files(self) -> bool:
        """
        Create ComfyUI configuration files.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create extra_model_paths.yaml to point to our models
            extra_paths_config = self.comfyui_path / "extra_model_paths.yaml"

            config_content = f"""# Extra model paths for ExcuseMyFrench
excusemyfrench:
    base_path: {self.wan_model_path.parent.absolute()}
    checkpoints: wan2.2/
    loras: lora/

# Output directory
output:
    base_path: {self.output_path.absolute()}
"""

            with open(extra_paths_config, 'w') as f:
                f.write(config_content)

            logger.info(f"Created config file: {extra_paths_config}")

            # Create output directory
            self.output_path.mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            logger.error(f"Failed to create config files: {e}")
            return False

    def create_startup_script(self) -> bool:
        """
        Create startup script for ComfyUI server.

        Returns:
            True if successful, False otherwise
        """
        try:
            if platform.system() == "Windows":
                script_name = "start_comfyui.bat"
                script_content = f"""@echo off
echo Starting ComfyUI server...
cd /d "{self.comfyui_path}"
python main.py --listen 127.0.0.1 --port 8188
"""
            else:
                script_name = "start_comfyui.sh"
                script_content = f"""#!/bin/bash
echo "Starting ComfyUI server..."
cd "{self.comfyui_path}"
python main.py --listen 127.0.0.1 --port 8188
"""

            script_path = self.comfyui_path / script_name
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable on Unix systems
            if platform.system() != "Windows":
                script_path.chmod(0o755)

            logger.info(f"Created startup script: {script_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to create startup script: {e}")
            return False

    def validate_installation(self) -> Tuple[bool, List[str]]:
        """
        Validate that ComfyUI is properly installed.

        Returns:
            Tuple of (success, list of issues)
        """
        issues = []

        # Check ComfyUI directory
        if not self.comfyui_path.exists():
            issues.append(f"ComfyUI directory not found: {self.comfyui_path}")
            return False, issues

        # Check main.py
        main_py = self.comfyui_path / "main.py"
        if not main_py.exists():
            issues.append("ComfyUI main.py not found")

        # Check models directory
        if not self.models_path.exists():
            issues.append("Models directory not found")

        # Check for at least one Wan model
        checkpoints_path = self.models_path / "checkpoints"
        if checkpoints_path.exists():
            wan_models = list(checkpoints_path.glob("wan_*.safetensors"))
            if not wan_models:
                issues.append("No Wan 2.2 models found in checkpoints")
        else:
            issues.append("Checkpoints directory not found")

        # Check custom nodes
        if not self.custom_nodes_path.exists():
            issues.append("Custom nodes directory not found")

        return len(issues) == 0, issues

    def run_setup(self) -> bool:
        """
        Run complete setup process.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info("ComfyUI Setup for ExcuseMyFrench")
        logger.info("=" * 60)

        # Check prerequisites
        logger.info("\n[1/8] Checking prerequisites...")
        prereq_ok, missing = self.check_prerequisites()
        if not prereq_ok:
            logger.error("Missing prerequisites:")
            for item in missing:
                logger.error(f"  - {item}")
            return False
        logger.info("All prerequisites met")

        # Clone ComfyUI
        logger.info("\n[2/8] Cloning ComfyUI repository...")
        if not self.clone_comfyui():
            return False

        # Install dependencies
        logger.info("\n[3/8] Installing dependencies...")
        if not self.install_dependencies():
            return False

        # Setup custom nodes
        logger.info("\n[4/8] Installing custom nodes...")
        if not self.setup_custom_nodes():
            return False

        # Download Wan models
        logger.info("\n[5/8] Downloading Wan 2.2 models...")
        logger.info("NOTE: Model downloads may take a long time depending on your connection")
        if not self.download_wan_models():
            logger.warning("Some models failed to download, but continuing...")

        # Create config files
        logger.info("\n[6/8] Creating configuration files...")
        if not self.create_config_files():
            return False

        # Create startup script
        logger.info("\n[7/8] Creating startup script...")
        if not self.create_startup_script():
            return False

        # Validate installation
        logger.info("\n[8/8] Validating installation...")
        valid, issues = self.validate_installation()
        if not valid:
            logger.error("Installation validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False

        logger.info("\n" + "=" * 60)
        logger.info("ComfyUI setup completed successfully!")
        logger.info("=" * 60)
        logger.info(f"\nInstallation directory: {self.comfyui_path}")
        logger.info(f"Models directory: {self.models_path}")
        logger.info(f"Output directory: {self.output_path}")
        logger.info(f"\nTo start ComfyUI server:")
        if platform.system() == "Windows":
            logger.info(f"  {self.comfyui_path / 'start_comfyui.bat'}")
        else:
            logger.info(f"  {self.comfyui_path / 'start_comfyui.sh'}")
        logger.info(f"\nServer will be available at: {self.server_url}")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup ComfyUI with Wan 2.2 models for ExcuseMyFrench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup with default settings
  python setup_comfyui.py

  # Setup with custom installation path
  python setup_comfyui.py --install-path /custom/path/ComfyUI

  # Force reinstallation
  python setup_comfyui.py --force

  # Verbose output
  python setup_comfyui.py --verbose
        """
    )

    parser.add_argument(
        "--install-path",
        help="Custom installation path for ComfyUI"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reinstallation even if already installed"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Run setup
    setup = ComfyUISetup(
        install_path=args.install_path,
        force=args.force
    )

    try:
        success = setup.run_setup()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
