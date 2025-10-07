#!/usr/bin/env python3
"""
Model integrity verification script.

Verifies that model files match expected checksums to detect:
- Corrupted downloads
- Tampered files
- Incomplete transfers

Usage:
    # Generate checksums for existing models
    python scripts/verify_models.py --generate

    # Verify models against checksums
    python scripts/verify_models.py --verify

    # Verify specific model
    python scripts/verify_models.py --verify --model wan_2.2_base.safetensors
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


class ModelVerifier:
    """Handles model file verification and checksum management."""

    def __init__(self, checksums_file: Optional[str] = None):
        """
        Initialize the ModelVerifier.

        Args:
            checksums_file: Path to checksums.json (default: models/checksums.json)
        """
        project_root = Path(__file__).parent.parent

        if checksums_file:
            self.checksums_file = Path(checksums_file)
        else:
            self.checksums_file = project_root / "models" / "checksums.json"

        self.project_root = project_root
        self.checksums_data = self._load_checksums()

    def _load_checksums(self) -> Dict:
        """Load checksums from JSON file."""
        if not self.checksums_file.exists():
            print(f"Warning: Checksums file not found: {self.checksums_file}")
            return {
                "version": "1.0.0",
                "models": {}
            }

        with open(self.checksums_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_checksums(self):
        """Save checksums to JSON file."""
        self.checksums_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checksums_file, 'w', encoding='utf-8') as f:
            json.dump(self.checksums_data, f, indent=2, ensure_ascii=False)

        print(f"Checksums saved to: {self.checksums_file}")

    def calculate_file_hash(
        self,
        file_path: Path,
        algorithm: str = "sha256",
        chunk_size: int = 8192
    ) -> Tuple[str, int]:
        """
        Calculate hash of a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm (default: sha256)
            chunk_size: Bytes to read at a time (for large files)

        Returns:
            Tuple of (hex_digest, file_size_bytes)
        """
        hash_obj = hashlib.new(algorithm)
        file_size = 0

        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_obj.update(chunk)
                file_size += len(chunk)

        return hash_obj.hexdigest(), file_size

    def generate_checksums(self, model_name: Optional[str] = None):
        """
        Generate checksums for model files.

        Args:
            model_name: Specific model to generate checksum for (default: all)
        """
        print("=" * 80)
        print("Generating Model Checksums")
        print("=" * 80)
        print()

        models = self.checksums_data.get("models", {})

        if model_name:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found in checksums.json")
                return False

            models_to_process = {model_name: models[model_name]}
        else:
            models_to_process = models

        updated_count = 0
        skipped_count = 0

        for name, info in models_to_process.items():
            model_path = self.project_root / info["path"]

            print(f"Processing: {name}")
            print(f"  Path: {model_path}")

            if not model_path.exists():
                print(f"  Status: NOT FOUND (skipping)")
                print()
                skipped_count += 1
                continue

            # Check if it's a directory (some models are collections of files)
            if model_path.is_dir():
                print(f"  Status: DIRECTORY (individual files not checksummed)")
                print(f"  Note: {info.get('notes', 'Contains multiple model files')}")
                print()
                skipped_count += 1
                continue

            # Calculate checksum
            try:
                print(f"  Calculating checksum...", end=" ", flush=True)
                checksum, file_size = self.calculate_file_hash(model_path)

                # Update checksums data
                info["sha256"] = checksum
                info["size_bytes"] = file_size
                info["verified"] = True

                size_mb = file_size / (1024 * 1024)
                print(f"DONE")
                print(f"  SHA256: {checksum}")
                print(f"  Size: {file_size:,} bytes ({size_mb:.2f} MB)")
                print()

                updated_count += 1

            except Exception as e:
                print(f"ERROR")
                print(f"  Error: {e}")
                print()
                continue

        # Save updated checksums
        if updated_count > 0:
            self._save_checksums()

        # Summary
        print("=" * 80)
        print(f"Summary: {updated_count} checksums generated, {skipped_count} skipped")
        print("=" * 80)
        print()

        return updated_count > 0

    def verify_models(self, model_name: Optional[str] = None) -> bool:
        """
        Verify model files against stored checksums.

        Args:
            model_name: Specific model to verify (default: all)

        Returns:
            True if all verifications passed, False otherwise
        """
        print("=" * 80)
        print("Verifying Model Integrity")
        print("=" * 80)
        print()

        models = self.checksums_data.get("models", {})

        if model_name:
            if model_name not in models:
                print(f"Error: Model '{model_name}' not found in checksums.json")
                return False

            models_to_verify = {model_name: models[model_name]}
        else:
            models_to_verify = models

        passed_count = 0
        failed_count = 0
        skipped_count = 0

        for name, info in models_to_verify.items():
            model_path = self.project_root / info["path"]
            expected_checksum = info.get("sha256")

            print(f"Verifying: {name}")
            print(f"  Path: {model_path}")

            # Check if file exists
            if not model_path.exists():
                if info.get("required", False):
                    print(f"  Status: ✗ MISSING (required)")
                    failed_count += 1
                else:
                    print(f"  Status: - NOT FOUND (optional)")
                    skipped_count += 1
                print()
                continue

            # Skip directories
            if model_path.is_dir():
                print(f"  Status: - DIRECTORY (skipped)")
                print()
                skipped_count += 1
                continue

            # Check if checksum exists
            if not expected_checksum:
                print(f"  Status: - NO CHECKSUM (run --generate first)")
                print()
                skipped_count += 1
                continue

            # Verify checksum
            try:
                print(f"  Calculating checksum...", end=" ", flush=True)
                actual_checksum, file_size = self.calculate_file_hash(model_path)

                # Check size
                expected_size = info.get("size_bytes")
                if expected_size and file_size != expected_size:
                    print(f"FAILED")
                    print(f"  Status: ✗ SIZE MISMATCH")
                    print(f"  Expected: {expected_size:,} bytes")
                    print(f"  Actual: {file_size:,} bytes")
                    failed_count += 1
                    print()
                    continue

                # Check checksum
                if actual_checksum != expected_checksum:
                    print(f"FAILED")
                    print(f"  Status: ✗ CHECKSUM MISMATCH")
                    print(f"  Expected: {expected_checksum}")
                    print(f"  Actual: {actual_checksum}")
                    print(f"  WARNING: File may be corrupted or tampered!")
                    failed_count += 1
                else:
                    print(f"PASSED")
                    print(f"  Status: ✓ VERIFIED")
                    print(f"  SHA256: {actual_checksum}")
                    size_mb = file_size / (1024 * 1024)
                    print(f"  Size: {file_size:,} bytes ({size_mb:.2f} MB)")
                    passed_count += 1

                print()

            except Exception as e:
                print(f"ERROR")
                print(f"  Status: ✗ VERIFICATION ERROR")
                print(f"  Error: {e}")
                failed_count += 1
                print()

        # Summary
        total = passed_count + failed_count + skipped_count
        print("=" * 80)
        print("Verification Summary")
        print("=" * 80)
        print(f"Total: {total} models")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Skipped: {skipped_count}")

        if failed_count == 0:
            print("\n✓ All verifications passed!")
        else:
            print(f"\n✗ {failed_count} verification(s) failed!")

        print("=" * 80)
        print()

        return failed_count == 0

    def list_models(self):
        """List all models in checksums.json."""
        print("=" * 80)
        print("Registered Models")
        print("=" * 80)
        print()

        models = self.checksums_data.get("models", {})

        if not models:
            print("No models registered in checksums.json")
            return

        for name, info in models.items():
            model_path = self.project_root / info["path"]
            exists = "✓" if model_path.exists() else "✗"
            verified = "✓" if info.get("verified", False) else "-"
            required = "Required" if info.get("required", False) else "Optional"

            print(f"Name: {name}")
            print(f"  Path: {info['path']}")
            print(f"  Exists: {exists}")
            print(f"  Verified: {verified}")
            print(f"  Status: {required}")
            print(f"  Description: {info.get('description', 'N/A')}")

            if info.get("sha256"):
                print(f"  SHA256: {info['sha256'][:16]}...")

            print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify model file integrity using checksums"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate checksums for existing model files"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model files against stored checksums"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all registered models"
    )

    parser.add_argument(
        "--model",
        help="Specific model name to process (from checksums.json)"
    )

    parser.add_argument(
        "--checksums-file",
        help="Path to checksums.json file (default: models/checksums.json)"
    )

    args = parser.parse_args()

    # Create verifier
    verifier = ModelVerifier(checksums_file=args.checksums_file)

    # Execute requested operation
    if args.list:
        verifier.list_models()
        return

    if args.generate:
        success = verifier.generate_checksums(model_name=args.model)
        sys.exit(0 if success else 1)

    elif args.verify:
        success = verifier.verify_models(model_name=args.model)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        print("\nError: Please specify --generate, --verify, or --list")
        sys.exit(1)


if __name__ == '__main__':
    main()
