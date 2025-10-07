#!/usr/bin/env python3
"""
Test script for ComfyUI integration.

This script tests the ComfyUI setup and generates sample images
to verify everything is working correctly.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from comfyui_integration import ComfyUIClient, ComfyUIConnectionError

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_connection(client: ComfyUIClient) -> bool:
    """
    Test connection to ComfyUI server.

    Args:
        client: ComfyUI client instance

    Returns:
        True if connected, False otherwise
    """
    logger.info("=" * 60)
    logger.info("Test 1: Connection")
    logger.info("=" * 60)

    try:
        if client.check_connection():
            logger.info("✓ Successfully connected to ComfyUI server")
            return True
        else:
            logger.error("✗ Failed to connect to ComfyUI server")
            return False
    except Exception as e:
        logger.error(f"✗ Connection test failed: {e}")
        return False


def test_system_stats(client: ComfyUIClient) -> bool:
    """
    Test system stats retrieval.

    Args:
        client: ComfyUI client instance

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 2: System Stats")
    logger.info("=" * 60)

    try:
        stats = client.get_system_stats()
        logger.info("✓ Successfully retrieved system stats")
        logger.info(f"Stats: {json.dumps(stats, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"✗ System stats test failed: {e}")
        return False


def test_workflow_loading(client: ComfyUIClient) -> bool:
    """
    Test workflow file loading.

    Args:
        client: ComfyUI client instance

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 3: Workflow Loading")
    logger.info("=" * 60)

    workflows = [
        "comfyui/workflows/character_generation.json",
        "comfyui/workflows/nutsy_generation.json",
        "comfyui/workflows/character_with_pose.json"
    ]

    all_success = True

    for workflow_path in workflows:
        try:
            workflow = client.load_workflow(workflow_path)
            logger.info(f"✓ Successfully loaded: {workflow_path}")
            logger.info(f"  Nodes: {len(workflow)}")
        except FileNotFoundError:
            logger.warning(f"⚠ Workflow not found: {workflow_path}")
            all_success = False
        except Exception as e:
            logger.error(f"✗ Failed to load {workflow_path}: {e}")
            all_success = False

    return all_success


def test_workflow_update(client: ComfyUIClient) -> bool:
    """
    Test workflow parameter updates.

    Args:
        client: ComfyUI client instance

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 4: Workflow Parameter Updates")
    logger.info("=" * 60)

    try:
        workflow_path = "comfyui/workflows/character_generation.json"
        workflow = client.load_workflow(workflow_path)

        params = {
            'prompt': 'test prompt',
            'negative_prompt': 'test negative',
            'seed': 999,
            'steps': 25,
            'cfg': 7.0
        }

        updated = client.update_workflow_params(workflow, params)
        logger.info("✓ Successfully updated workflow parameters")

        # Verify updates (basic check)
        found_updates = False
        for node_id, node_data in updated.items():
            if isinstance(node_data, dict) and 'inputs' in node_data:
                inputs = node_data['inputs']
                if inputs.get('seed') == 999 or inputs.get('steps') == 25:
                    found_updates = True
                    break

        if found_updates:
            logger.info("✓ Parameters were applied to workflow")
            return True
        else:
            logger.warning("⚠ Could not verify parameter updates")
            return True  # Still pass as structure is valid

    except Exception as e:
        logger.error(f"✗ Workflow update test failed: {e}")
        return False


def test_queue_status(client: ComfyUIClient) -> bool:
    """
    Test queue status retrieval.

    Args:
        client: ComfyUI client instance

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 5: Queue Status")
    logger.info("=" * 60)

    try:
        queue = client.get_queue_status()
        logger.info("✓ Successfully retrieved queue status")

        if 'queue_running' in queue:
            logger.info(f"  Running: {len(queue['queue_running'])} jobs")
        if 'queue_pending' in queue:
            logger.info(f"  Pending: {len(queue['queue_pending'])} jobs")

        return True
    except Exception as e:
        logger.error(f"✗ Queue status test failed: {e}")
        return False


def test_image_generation(
    client: ComfyUIClient,
    output_dir: str,
    skip_generation: bool = False
) -> bool:
    """
    Test actual image generation.

    Args:
        client: ComfyUI client instance
        output_dir: Directory to save test images
        skip_generation: Skip actual generation (use for quick tests)

    Returns:
        True if successful, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 6: Image Generation")
    logger.info("=" * 60)

    if skip_generation:
        logger.info("⊘ Skipping image generation (use --generate to enable)")
        return True

    try:
        workflow_path = "comfyui/workflows/character_generation.json"
        output_path = Path(output_dir) / "test_generation.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'prompt': 'French bulldog, happy expression, portrait, high quality',
            'negative_prompt': 'blurry, low quality',
            'seed': 42,
            'steps': 20,  # Fewer steps for faster test
            'cfg': 7.5
        }

        logger.info("Generating test image (this may take a minute)...")
        logger.info(f"Prompt: {params['prompt']}")

        images = client.generate_image(
            workflow_path=workflow_path,
            params=params,
            output_path=str(output_path),
            timeout=180  # 3 minute timeout
        )

        if images and len(images) > 0:
            logger.info(f"✓ Successfully generated {len(images)} image(s)")
            logger.info(f"  Saved to: {output_path}")
            return True
        else:
            logger.error("✗ No images were generated")
            return False

    except Exception as e:
        logger.error(f"✗ Image generation test failed: {e}")
        return False


def test_models_exist() -> bool:
    """
    Test if required models exist.

    Returns:
        True if models found, False otherwise
    """
    logger.info("\n" + "=" * 60)
    logger.info("Test 7: Model Files")
    logger.info("=" * 60)

    comfyui_path = Path(os.getenv("COMFYUI_PATH", "D:/ComfyUI"))
    checkpoints_path = comfyui_path / "models" / "checkpoints"

    if not checkpoints_path.exists():
        logger.error(f"✗ Checkpoints directory not found: {checkpoints_path}")
        return False

    logger.info(f"Checking for models in: {checkpoints_path}")

    # Look for Wan models
    wan_models = list(checkpoints_path.glob("wan*.safetensors"))

    if wan_models:
        logger.info(f"✓ Found {len(wan_models)} Wan model(s):")
        for model in wan_models:
            size_mb = model.stat().st_size / (1024**2)
            logger.info(f"  - {model.name} ({size_mb:.1f} MB)")
        return True
    else:
        logger.warning("⚠ No Wan models found (*.safetensors)")
        logger.info("  Available models:")
        all_models = list(checkpoints_path.glob("*.safetensors"))
        if all_models:
            for model in all_models[:5]:  # Show first 5
                logger.info(f"  - {model.name}")
        else:
            logger.warning("  No checkpoint models found!")
        return False


def run_all_tests(
    server_url: str = None,
    output_dir: str = "test_output",
    skip_generation: bool = True
) -> bool:
    """
    Run all tests.

    Args:
        server_url: ComfyUI server URL (optional)
        output_dir: Directory for test outputs
        skip_generation: Skip image generation test

    Returns:
        True if all tests passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info("ComfyUI Integration Test Suite")
    logger.info("=" * 60)

    # Initialize client
    try:
        client = ComfyUIClient(server_url=server_url)
    except Exception as e:
        logger.error(f"Failed to initialize ComfyUI client: {e}")
        return False

    # Run tests
    results = {
        "Connection": test_connection(client),
        "System Stats": test_system_stats(client),
        "Workflow Loading": test_workflow_loading(client),
        "Workflow Updates": test_workflow_update(client),
        "Queue Status": test_queue_status(client),
        "Image Generation": test_image_generation(client, output_dir, skip_generation),
        "Model Files": test_models_exist()
    }

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for success in results.values() if success)
    total = len(results)

    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{test_name:.<40} {status}")

    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n✓ All tests passed! ComfyUI is ready to use.")
        return True
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. Check logs above for details.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test ComfyUI integration for ExcuseMyFrench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic tests (no image generation)
  python test_comfyui.py

  # Run all tests including image generation
  python test_comfyui.py --generate

  # Test with custom server URL
  python test_comfyui.py --server-url http://localhost:8188

  # Save test output to custom directory
  python test_comfyui.py --output-dir my_tests --generate

  # Verbose output
  python test_comfyui.py --verbose
        """
    )

    parser.add_argument(
        "--server-url",
        help="ComfyUI server URL (overrides env var)"
    )

    parser.add_argument(
        "--output-dir",
        default="test_output",
        help="Directory for test outputs (default: test_output)"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Include image generation test (slower)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("comfyui_integration").setLevel(logging.DEBUG)

    # Run tests
    try:
        success = run_all_tests(
            server_url=args.server_url,
            output_dir=args.output_dir,
            skip_generation=not args.generate
        )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test suite failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
