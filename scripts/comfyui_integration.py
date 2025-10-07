#!/usr/bin/env python3
"""
ComfyUI API integration for ExcuseMyFrench.

This module provides a client for interacting with the ComfyUI API,
managing image generation workflows, and handling batch operations.
"""

import argparse
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, parse
import io

from PIL import Image
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)

# Import image cache
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.image_cache import ImageCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ComfyUIError(Exception):
    """Base exception for ComfyUI errors."""
    pass


class ComfyUIConnectionError(ComfyUIError):
    """Raised when connection to ComfyUI fails."""
    pass


class ComfyUIWorkflowError(ComfyUIError):
    """Raised when workflow execution fails."""
    pass


class ComfyUIClient:
    """Client for interacting with ComfyUI API."""

    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout: int = 300,
        max_retries: int = 3,
        use_cache: bool = None
    ):
        """
        Initialize ComfyUI client.

        Args:
            server_url: ComfyUI server URL (uses env var if None)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            use_cache: Enable image caching (None = use env var)
        """
        if server_url:
            self.server_url = server_url.rstrip('/')
        else:
            self.server_url = os.getenv(
                "COMFYUI_SERVER_URL",
                "http://127.0.0.1:8188"
            ).rstrip('/')

        self.timeout = timeout
        self.max_retries = max_retries
        self.client_id = str(uuid.uuid4())

        # Initialize cache
        if use_cache is None:
            use_cache = os.getenv("COMFYUI_CACHE_ENABLED", "true").lower() == "true"

        self.cache = ImageCache() if use_cache else None

        logger.info(f"ComfyUI client initialized")
        logger.info(f"Server URL: {self.server_url}")
        logger.info(f"Client ID: {self.client_id}")
        if self.cache:
            logger.info(f"Image caching enabled")

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        retry: int = 0
    ) -> Any:
        """
        Make HTTP request to ComfyUI API.

        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, etc.)
            data: Request data for POST requests
            retry: Current retry attempt

        Returns:
            Response data

        Raises:
            ComfyUIConnectionError: If connection fails
        """
        url = f"{self.server_url}/{endpoint.lstrip('/')}"

        try:
            if method == "GET":
                req = request.Request(url, method=method)
            elif method == "POST":
                json_data = json.dumps(data).encode('utf-8')
                req = request.Request(
                    url,
                    data=json_data,
                    method=method,
                    headers={'Content-Type': 'application/json'}
                )
            else:
                raise ValueError(f"Unsupported method: {method}")

            with request.urlopen(req, timeout=self.timeout) as response:
                if response.status == 200:
                    return json.loads(response.read().decode('utf-8'))
                else:
                    raise ComfyUIConnectionError(
                        f"Request failed with status {response.status}"
                    )

        except Exception as e:
            if retry < self.max_retries:
                wait_time = 2 ** retry  # Exponential backoff
                logger.warning(f"Request failed, retrying in {wait_time}s... ({retry + 1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._request(endpoint, method, data, retry + 1)
            else:
                raise ComfyUIConnectionError(f"Failed to connect to ComfyUI: {e}") from e

    def check_connection(self) -> bool:
        """
        Check if ComfyUI server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            response = self._request("/system_stats")
            logger.info("Successfully connected to ComfyUI server")
            return True
        except Exception as e:
            logger.error(f"Cannot connect to ComfyUI server: {e}")
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get ComfyUI system statistics.

        Returns:
            Dictionary with system stats
        """
        return self._request("/system_stats")

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status.

        Returns:
            Dictionary with queue information
        """
        return self._request("/queue")

    def load_workflow(self, workflow_path: str) -> Dict[str, Any]:
        """
        Load workflow from JSON file.

        Args:
            workflow_path: Path to workflow JSON file

        Returns:
            Workflow dictionary

        Raises:
            FileNotFoundError: If workflow file doesn't exist
            json.JSONDecodeError: If workflow is not valid JSON
        """
        workflow_file = Path(workflow_path)

        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")

        with open(workflow_file, 'r') as f:
            workflow = json.load(f)

        logger.info(f"Loaded workflow from {workflow_path}")
        return workflow

    def update_workflow_params(
        self,
        workflow: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update workflow parameters.

        Args:
            workflow: Workflow dictionary
            params: Parameters to update

        Returns:
            Updated workflow dictionary
        """
        # Make a copy to avoid modifying original
        updated_workflow = json.loads(json.dumps(workflow))

        # Common parameter mappings
        param_mappings = {
            'prompt': 'text',
            'negative_prompt': 'text',
            'seed': 'seed',
            'steps': 'steps',
            'cfg': 'cfg',
            'width': 'width',
            'height': 'height',
            'batch_size': 'batch_size',
            'model': 'ckpt_name'
        }

        # Update workflow nodes with provided parameters
        for node_id, node_data in updated_workflow.items():
            if not isinstance(node_data, dict):
                continue

            inputs = node_data.get('inputs', {})

            # Update based on node class type
            class_type = node_data.get('class_type', '')

            # KSampler nodes
            if 'Sampler' in class_type or 'KSampler' in class_type:
                if 'seed' in params:
                    inputs['seed'] = params['seed']
                if 'steps' in params:
                    inputs['steps'] = params['steps']
                if 'cfg' in params:
                    inputs['cfg'] = params['cfg']

            # Text prompt nodes
            elif 'CLIP' in class_type and 'Text' in class_type:
                if 'prompt' in params and 'positive' in str(node_id).lower():
                    inputs['text'] = params['prompt']
                if 'negative_prompt' in params and 'negative' in str(node_id).lower():
                    inputs['text'] = params['negative_prompt']

            # Checkpoint loader nodes
            elif 'CheckpointLoader' in class_type:
                if 'model' in params:
                    inputs['ckpt_name'] = params['model']

            # Empty latent image nodes
            elif 'EmptyLatentImage' in class_type:
                if 'width' in params:
                    inputs['width'] = params['width']
                if 'height' in params:
                    inputs['height'] = params['height']
                if 'batch_size' in params:
                    inputs['batch_size'] = params['batch_size']

            node_data['inputs'] = inputs

        return updated_workflow

    def queue_prompt(self, workflow: Dict[str, Any]) -> str:
        """
        Queue a workflow for execution.

        Args:
            workflow: Workflow dictionary

        Returns:
            Prompt ID

        Raises:
            ComfyUIWorkflowError: If queueing fails
        """
        try:
            data = {
                "prompt": workflow,
                "client_id": self.client_id
            }

            response = self._request("/prompt", method="POST", data=data)

            prompt_id = response.get('prompt_id')
            if not prompt_id:
                raise ComfyUIWorkflowError("No prompt_id in response")

            logger.info(f"Queued workflow with prompt_id: {prompt_id}")
            return prompt_id

        except Exception as e:
            raise ComfyUIWorkflowError(f"Failed to queue workflow: {e}") from e

    def get_history(self, prompt_id: str) -> Dict[str, Any]:
        """
        Get execution history for a prompt.

        Args:
            prompt_id: Prompt ID

        Returns:
            History dictionary
        """
        return self._request(f"/history/{prompt_id}")

    def wait_for_completion(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Wait for workflow execution to complete.

        Args:
            prompt_id: Prompt ID to wait for
            poll_interval: Time between status checks in seconds
            timeout: Maximum wait time in seconds (None for no timeout)

        Returns:
            Execution history

        Raises:
            ComfyUIWorkflowError: If execution fails or times out
        """
        start_time = time.time()

        logger.info(f"Waiting for workflow completion: {prompt_id}")

        while True:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise ComfyUIWorkflowError(
                    f"Workflow execution timed out after {timeout}s"
                )

            # Get history
            history = self.get_history(prompt_id)

            if prompt_id in history:
                prompt_history = history[prompt_id]

                # Check for errors
                if 'error' in prompt_history:
                    error = prompt_history['error']
                    raise ComfyUIWorkflowError(f"Workflow execution failed: {error}")

                # Check if complete
                if 'outputs' in prompt_history:
                    elapsed = time.time() - start_time
                    logger.info(f"Workflow completed in {elapsed:.1f}s")
                    return prompt_history

            # Wait before next check
            time.sleep(poll_interval)

    def get_output_images(self, prompt_history: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract output images from prompt history.

        Args:
            prompt_history: Prompt execution history

        Returns:
            List of image info dictionaries
        """
        images = []

        outputs = prompt_history.get('outputs', {})

        for node_id, node_output in outputs.items():
            if 'images' in node_output:
                for img_data in node_output['images']:
                    images.append({
                        'filename': img_data['filename'],
                        'subfolder': img_data.get('subfolder', ''),
                        'type': img_data.get('type', 'output'),
                        'node_id': node_id
                    })

        logger.info(f"Found {len(images)} output images")
        return images

    def download_image(
        self,
        image_info: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Download an output image.

        Args:
            image_info: Image info dictionary from get_output_images
            output_path: Optional path to save image

        Returns:
            PIL Image object

        Raises:
            ComfyUIError: If download fails
        """
        try:
            params = {
                'filename': image_info['filename'],
                'subfolder': image_info.get('subfolder', ''),
                'type': image_info.get('type', 'output')
            }

            url = f"{self.server_url}/view?{parse.urlencode(params)}"

            with request.urlopen(url, timeout=self.timeout) as response:
                image_data = response.read()

            image = Image.open(io.BytesIO(image_data))

            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_file)
                logger.info(f"Saved image to {output_path}")

            return image

        except Exception as e:
            raise ComfyUIError(f"Failed to download image: {e}") from e

    def generate_image(
        self,
        workflow_path: str,
        params: Dict[str, Any],
        output_path: Optional[str] = None,
        timeout: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Image.Image]:
        """
        Generate images using a workflow.

        Args:
            workflow_path: Path to workflow JSON file
            params: Parameters to update in workflow
            output_path: Directory to save images (optional)
            timeout: Maximum wait time in seconds
            use_cache: Check cache before generating (default: True)

        Returns:
            List of generated PIL Images

        Raises:
            ComfyUIError: If generation fails
        """
        # Create cache key from params and workflow
        cache_params = {
            'workflow': workflow_path,
            'params': params
        }

        # Try to get from cache first
        if use_cache and self.cache:
            cached_image = self.cache.get(cache_params)
            if cached_image:
                logger.info("Using cached image")
                images = [cached_image]

                # Save to output path if requested
                if output_path:
                    cached_image.save(output_path)
                    logger.info(f"Saved cached image to {output_path}")

                return images

        # Not in cache, generate new image
        logger.info("Generating new image (cache miss or disabled)")

        # Load workflow
        workflow = self.load_workflow(workflow_path)

        # Update parameters
        workflow = self.update_workflow_params(workflow, params)

        # Queue workflow
        prompt_id = self.queue_prompt(workflow)

        # Wait for completion
        history = self.wait_for_completion(prompt_id, timeout=timeout)

        # Get output images
        image_infos = self.get_output_images(history)

        # Download images
        images = []
        for i, img_info in enumerate(image_infos):
            if output_path:
                # Create filename based on parameters
                basename = Path(output_path).stem
                ext = Path(output_path).suffix or '.png'
                if len(image_infos) > 1:
                    save_path = f"{basename}_{i}{ext}"
                else:
                    save_path = output_path
            else:
                save_path = None

            image = self.download_image(img_info, save_path)
            images.append(image)

        # Cache the first image (if caching enabled)
        if use_cache and self.cache and images:
            self.cache.put(cache_params, images[0])

        logger.info(f"Generated {len(images)} images")
        return images

    def batch_generate(
        self,
        workflow_path: str,
        param_list: List[Dict[str, Any]],
        output_dir: str,
        max_concurrent: int = 3
    ) -> List[Tuple[Dict[str, Any], List[str]]]:
        """
        Generate multiple images in batch.

        Args:
            workflow_path: Path to workflow JSON file
            param_list: List of parameter dictionaries
            output_dir: Directory to save images
            max_concurrent: Maximum concurrent jobs (not currently enforced)

        Returns:
            List of (params, output_paths) tuples

        Note:
            Currently processes sequentially. Concurrent processing would
            require WebSocket support for real-time queue monitoring.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []

        logger.info(f"Starting batch generation of {len(param_list)} images")

        for i, params in enumerate(param_list, 1):
            logger.info(f"Processing {i}/{len(param_list)}")

            try:
                # Generate unique filename
                character = params.get('character', 'unknown')
                emotion = params.get('emotion', 'neutral')
                timestamp = int(time.time() * 1000)
                filename = f"{character}_{emotion}_{timestamp}.png"
                save_path = str(output_path / filename)

                # Generate image
                images = self.generate_image(
                    workflow_path=workflow_path,
                    params=params,
                    output_path=save_path
                )

                results.append((params, [save_path]))

            except Exception as e:
                logger.error(f"Failed to generate image {i}/{len(param_list)}: {e}")
                results.append((params, []))
                continue

        success_count = sum(1 for _, paths in results if paths)
        logger.info(f"Batch generation complete: {success_count}/{len(param_list)} successful")

        return results

    def interrupt_execution(self) -> bool:
        """
        Interrupt current workflow execution.

        Returns:
            True if successful
        """
        try:
            self._request("/interrupt", method="POST", data={})
            logger.info("Execution interrupted")
            return True
        except Exception as e:
            logger.error(f"Failed to interrupt execution: {e}")
            return False

    def clear_queue(self) -> bool:
        """
        Clear the execution queue.

        Returns:
            True if successful
        """
        try:
            data = {"clear": True}
            self._request("/queue", method="POST", data=data)
            logger.info("Queue cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            return False


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(
        description="ComfyUI API client for ExcuseMyFrench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check connection
  python comfyui_integration.py --check

  # Generate single image
  python comfyui_integration.py --workflow comfyui/workflows/character_generation.json --prompt "French bulldog, happy" --output output.png

  # Get system stats
  python comfyui_integration.py --stats

  # Get queue status
  python comfyui_integration.py --queue
        """
    )

    parser.add_argument(
        "--server-url",
        help="ComfyUI server URL (overrides env var)"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check connection to ComfyUI server"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Get system statistics"
    )

    parser.add_argument(
        "--queue",
        action="store_true",
        help="Get queue status"
    )

    parser.add_argument(
        "--workflow",
        help="Path to workflow JSON file"
    )

    parser.add_argument(
        "--prompt",
        help="Prompt text for image generation"
    )

    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt text"
    )

    parser.add_argument(
        "--output",
        help="Output image path"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize client
    try:
        client = ComfyUIClient(server_url=args.server_url)
    except Exception as e:
        logger.error(f"Failed to initialize client: {e}")
        return 1

    # Execute commands
    if args.check:
        if client.check_connection():
            print("Connection successful")
            return 0
        else:
            print("Connection failed")
            return 1

    elif args.stats:
        try:
            stats = client.get_system_stats()
            print(json.dumps(stats, indent=2))
            return 0
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return 1

    elif args.queue:
        try:
            queue = client.get_queue_status()
            print(json.dumps(queue, indent=2))
            return 0
        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return 1

    elif args.workflow:
        if not args.prompt:
            logger.error("--prompt required for image generation")
            return 1

        try:
            params = {
                'prompt': args.prompt,
                'negative_prompt': args.negative_prompt
            }

            images = client.generate_image(
                workflow_path=args.workflow,
                params=params,
                output_path=args.output
            )

            print(f"Generated {len(images)} images")
            return 0

        except Exception as e:
            logger.error(f"Failed to generate image: {e}")
            return 1

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
