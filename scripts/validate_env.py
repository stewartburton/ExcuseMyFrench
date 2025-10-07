#!/usr/bin/env python3
"""
Environment validation script for ExcuseMyFrench project.

Validates that all required environment variables and dependencies are properly
configured before running the application. This helps catch configuration issues
early.

Usage:
    python scripts/validate_env.py
    python scripts/validate_env.py --strict  # Exit on any warning
    python scripts/validate_env.py --fix     # Attempt to fix issues
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    # Load environment from config/.env
    env_path = Path(__file__).parent.parent / "config" / ".env"
    load_dotenv(env_path)
except ImportError:
    print("Warning: python-dotenv not installed. Cannot load .env file.")


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, passed: bool, message: str, severity: str = "error"):
        self.passed = passed
        self.message = message
        self.severity = severity  # "error", "warning", "info"


class EnvironmentValidator:
    """Validates environment configuration."""

    # Required environment variables
    REQUIRED_VARS = {
        # API Keys (at least one LLM provider required)
        "ANTHROPIC_API_KEY": {
            "description": "Anthropic Claude API key",
            "required": False,  # Optional if OpenAI is configured
            "example": "sk-ant-...",
            "group": "llm"
        },
        "OPENAI_API_KEY": {
            "description": "OpenAI API key",
            "required": False,  # Optional if Anthropic is configured
            "example": "sk-...",
            "group": "llm"
        },
    }

    # Optional but recommended environment variables
    RECOMMENDED_VARS = {
        "META_ACCESS_TOKEN": {
            "description": "Meta/Facebook access token for Instagram API",
            "example": "EAA...",
        },
        "INSTAGRAM_USER_ID": {
            "description": "Instagram user ID for posting",
            "example": "1234567890",
        },
        "ELEVENLABS_API_KEY": {
            "description": "ElevenLabs API key for voice generation",
            "example": "...",
        },
    }

    # System dependencies
    SYSTEM_DEPENDENCIES = {
        "ffmpeg": {
            "command": ["ffmpeg", "-version"],
            "required": True,
            "description": "FFmpeg for video processing"
        },
        "ffprobe": {
            "command": ["ffprobe", "-version"],
            "required": True,
            "description": "FFprobe for video analysis"
        },
        "git": {
            "command": ["git", "--version"],
            "required": False,
            "description": "Git for version control"
        },
    }

    # Python package dependencies
    PYTHON_DEPENDENCIES = [
        "anthropic",
        "openai",
        "requests",
        "python-dotenv",
        "pillow",
        "ffmpeg-python",
        "pytrends",
    ]

    # Required directories
    REQUIRED_DIRECTORIES = [
        "data",
        "data/scripts",
        "data/audio",
        "data/images",
        "data/videos",
        "data/animated",
        "data/final_videos",
        "models",
        "config",
    ]

    def __init__(self, strict: bool = False, verbose: bool = False):
        """
        Initialize validator.

        Args:
            strict: Treat warnings as errors
            verbose: Show detailed output
        """
        self.strict = strict
        self.verbose = verbose
        self.results: List[ValidationResult] = []

    def validate_all(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all validations passed, False otherwise
        """
        print("=" * 80)
        print("ExcuseMyFrench Environment Validation")
        print("=" * 80)
        print()

        # Run all validation checks
        self.validate_environment_variables()
        self.validate_llm_providers()
        self.validate_system_dependencies()
        self.validate_python_dependencies()
        self.validate_directories()
        self.validate_config_file()

        # Print summary
        self._print_summary()

        # Determine overall result
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]

        if errors:
            return False

        if self.strict and warnings:
            return False

        return True

    def validate_environment_variables(self):
        """Validate required environment variables."""
        print("Checking Environment Variables...")
        print("-" * 80)

        for var_name, var_info in self.REQUIRED_VARS.items():
            value = os.getenv(var_name)

            if not value:
                if var_info.get("required", True):
                    self.results.append(ValidationResult(
                        False,
                        f"Required variable {var_name} not set. {var_info['description']}",
                        "error"
                    ))
                else:
                    self.results.append(ValidationResult(
                        True,
                        f"Optional variable {var_name} not set",
                        "info"
                    ))
            else:
                # Check if it looks like a placeholder
                if value.startswith("your-") or value == "...":
                    self.results.append(ValidationResult(
                        False,
                        f"{var_name} appears to be a placeholder value",
                        "warning"
                    ))
                else:
                    if self.verbose:
                        masked_value = value[:8] + "..." if len(value) > 8 else "***"
                        self.results.append(ValidationResult(
                            True,
                            f"{var_name} is set ({masked_value})",
                            "info"
                        ))

        # Check recommended variables
        for var_name, var_info in self.RECOMMENDED_VARS.items():
            value = os.getenv(var_name)

            if not value:
                self.results.append(ValidationResult(
                    False,
                    f"Recommended variable {var_name} not set. {var_info['description']}",
                    "warning"
                ))

        print()

    def validate_llm_providers(self):
        """Validate that at least one LLM provider is configured."""
        print("Checking LLM Providers...")
        print("-" * 80)

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        # Check if at least one is configured
        if not anthropic_key and not openai_key:
            self.results.append(ValidationResult(
                False,
                "No LLM provider configured. Set either ANTHROPIC_API_KEY or OPENAI_API_KEY",
                "error"
            ))
        else:
            providers = []
            if anthropic_key and not anthropic_key.startswith("your-"):
                providers.append("Anthropic Claude")
            if openai_key and not openai_key.startswith("your-"):
                providers.append("OpenAI GPT")

            if providers:
                self.results.append(ValidationResult(
                    True,
                    f"LLM providers configured: {', '.join(providers)}",
                    "info"
                ))

        print()

    def validate_system_dependencies(self):
        """Validate system-level dependencies."""
        print("Checking System Dependencies...")
        print("-" * 80)

        for name, info in self.SYSTEM_DEPENDENCIES.items():
            try:
                result = subprocess.run(
                    info["command"],
                    capture_output=True,
                    timeout=5
                )

                if result.returncode == 0:
                    version_output = result.stdout.decode('utf-8', errors='ignore').split('\n')[0]
                    if self.verbose:
                        self.results.append(ValidationResult(
                            True,
                            f"{name} found: {version_output}",
                            "info"
                        ))
                else:
                    severity = "error" if info.get("required", True) else "warning"
                    self.results.append(ValidationResult(
                        False,
                        f"{name} not working properly. {info['description']}",
                        severity
                    ))

            except (FileNotFoundError, subprocess.TimeoutExpired):
                severity = "error" if info.get("required", True) else "warning"
                self.results.append(ValidationResult(
                    False,
                    f"{name} not found. {info['description']}",
                    severity
                ))

        print()

    def validate_python_dependencies(self):
        """Validate Python package dependencies."""
        print("Checking Python Dependencies...")
        print("-" * 80)

        for package in self.PYTHON_DEPENDENCIES:
            try:
                __import__(package.replace("-", "_"))
                if self.verbose:
                    self.results.append(ValidationResult(
                        True,
                        f"Python package '{package}' installed",
                        "info"
                    ))
            except ImportError:
                self.results.append(ValidationResult(
                    False,
                    f"Python package '{package}' not installed. Run: pip install {package}",
                    "warning"
                ))

        print()

    def validate_directories(self):
        """Validate required directory structure."""
        print("Checking Directory Structure...")
        print("-" * 80)

        project_root = Path(__file__).parent.parent

        for dir_path in self.REQUIRED_DIRECTORIES:
            full_path = project_root / dir_path

            if not full_path.exists():
                self.results.append(ValidationResult(
                    False,
                    f"Required directory missing: {dir_path}",
                    "warning"
                ))
            elif not full_path.is_dir():
                self.results.append(ValidationResult(
                    False,
                    f"Path exists but is not a directory: {dir_path}",
                    "error"
                ))
            else:
                if self.verbose:
                    self.results.append(ValidationResult(
                        True,
                        f"Directory exists: {dir_path}",
                        "info"
                    ))

        print()

    def validate_config_file(self):
        """Validate config file exists and is readable."""
        print("Checking Configuration File...")
        print("-" * 80)

        config_path = Path(__file__).parent.parent / "config" / ".env"

        if not config_path.exists():
            self.results.append(ValidationResult(
                False,
                f"Config file not found: {config_path}",
                "error"
            ))
        elif not config_path.is_file():
            self.results.append(ValidationResult(
                False,
                f"Config path exists but is not a file: {config_path}",
                "error"
            ))
        else:
            # Check if readable
            try:
                with open(config_path, 'r') as f:
                    lines = f.readlines()

                self.results.append(ValidationResult(
                    True,
                    f"Config file found with {len(lines)} lines",
                    "info"
                ))

                # Check for common issues
                if len(lines) == 0:
                    self.results.append(ValidationResult(
                        False,
                        "Config file is empty",
                        "warning"
                    ))

            except Exception as e:
                self.results.append(ValidationResult(
                    False,
                    f"Cannot read config file: {e}",
                    "error"
                ))

        print()

    def _print_summary(self):
        """Print validation summary."""
        print("=" * 80)
        print("Validation Summary")
        print("=" * 80)

        # Count results by severity
        errors = [r for r in self.results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.results if not r.passed and r.severity == "warning"]
        info = [r for r in self.results if r.passed and r.severity == "info"]

        # Print errors
        if errors:
            print(f"\n{len(errors)} ERROR(S):")
            for result in errors:
                print(f"  ✗ {result.message}")

        # Print warnings
        if warnings:
            print(f"\n{len(warnings)} WARNING(S):")
            for result in warnings:
                print(f"  ⚠ {result.message}")

        # Print info if verbose
        if self.verbose and info:
            print(f"\n{len(info)} PASSED:")
            for result in info:
                print(f"  ✓ {result.message}")

        # Overall status
        print("\n" + "=" * 80)
        if not errors and not warnings:
            print("✓ All checks passed! Environment is properly configured.")
        elif not errors:
            print("⚠ Passed with warnings. Some optional features may not work.")
        else:
            print("✗ Validation failed. Please fix the errors above.")

        print("=" * 80)
        print()

    def fix_issues(self):
        """Attempt to auto-fix some common issues."""
        print("Attempting to fix issues...")
        print("-" * 80)

        project_root = Path(__file__).parent.parent

        # Create missing directories
        for dir_path in self.REQUIRED_DIRECTORIES:
            full_path = project_root / dir_path
            if not full_path.exists():
                try:
                    full_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ Created directory: {dir_path}")
                except Exception as e:
                    print(f"✗ Failed to create directory {dir_path}: {e}")

        # Create .env file if missing
        config_path = project_root / "config" / ".env"
        if not config_path.exists():
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    f.write("# ExcuseMyFrench Configuration\n")
                    f.write("# Add your API keys and configuration here\n\n")
                    for var_name, var_info in {**self.REQUIRED_VARS, **self.RECOMMENDED_VARS}.items():
                        f.write(f"# {var_info['description']}\n")
                        if 'example' in var_info:
                            f.write(f"# Example: {var_info['example']}\n")
                        f.write(f"{var_name}=\n\n")

                print(f"✓ Created template config file: {config_path}")
                print("  Please edit this file and add your API keys")
            except Exception as e:
                print(f"✗ Failed to create config file: {e}")

        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate ExcuseMyFrench environment configuration"
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to auto-fix issues"
    )

    args = parser.parse_args()

    validator = EnvironmentValidator(strict=args.strict, verbose=args.verbose)

    if args.fix:
        validator.fix_issues()

    success = validator.validate_all()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
