#!/usr/bin/env python3
"""
Model manager for preloading and managing AI models.

This module provides a singleton model manager that keeps models in memory
for long-running processes, avoiding repeated model loading overhead.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton manager for AI models with lazy loading and memory management.

    This class maintains a single instance across the application lifecycle,
    allowing models to be loaded once and reused across multiple calls.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model manager."""
        if self._initialized:
            return

        self.enabled = os.getenv("MODEL_PRELOAD_ENABLED", "false").lower() == "true"
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_lock = Lock()

        logger.info(f"Model manager initialized (preload enabled: {self.enabled})")
        self._initialized = True

    def get_model(self, model_name: str, loader_func: callable, **kwargs) -> Any:
        """
        Get a model, loading it if necessary.

        Args:
            model_name: Unique identifier for the model
            loader_func: Function to load the model if not cached
            **kwargs: Arguments to pass to loader_func

        Returns:
            The loaded model
        """
        # If preloading disabled, always load fresh
        if not self.enabled:
            logger.debug(f"Preloading disabled, loading {model_name} fresh")
            return loader_func(**kwargs)

        # Check if model is already loaded
        if model_name in self.models:
            logger.debug(f"Using cached model: {model_name}")
            self._update_access_time(model_name)
            return self.models[model_name]

        # Load model (with lock to prevent duplicate loading)
        with self._load_lock:
            # Double-check after acquiring lock
            if model_name in self.models:
                logger.debug(f"Using cached model: {model_name}")
                self._update_access_time(model_name)
                return self.models[model_name]

            # Load the model
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()

            try:
                model = loader_func(**kwargs)
                load_time = time.time() - start_time

                # Cache the model
                self.models[model_name] = model
                self.model_metadata[model_name] = {
                    'loaded_at': time.time(),
                    'last_accessed': time.time(),
                    'load_time': load_time,
                    'access_count': 0
                }

                logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
                return model

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    def _update_access_time(self, model_name: str):
        """Update the last access time for a model."""
        if model_name in self.model_metadata:
            metadata = self.model_metadata[model_name]
            metadata['last_accessed'] = time.time()
            metadata['access_count'] = metadata.get('access_count', 0) + 1

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if model was unloaded, False if not found
        """
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            del self.models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            return True
        return False

    def unload_unused(self, max_idle_seconds: int = 3600) -> int:
        """
        Unload models that haven't been accessed recently.

        Args:
            max_idle_seconds: Maximum idle time before unloading (default: 1 hour)

        Returns:
            Number of models unloaded
        """
        current_time = time.time()
        to_unload = []

        for model_name, metadata in self.model_metadata.items():
            idle_time = current_time - metadata['last_accessed']
            if idle_time > max_idle_seconds:
                to_unload.append(model_name)

        for model_name in to_unload:
            self.unload_model(model_name)

        if to_unload:
            logger.info(f"Unloaded {len(to_unload)} unused models")

        return len(to_unload)

    def clear_all(self):
        """Clear all loaded models from memory."""
        count = len(self.models)
        self.models.clear()
        self.model_metadata.clear()
        logger.info(f"Cleared all {count} models from memory")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded models.

        Returns:
            Dictionary with model statistics
        """
        stats = {
            'enabled': self.enabled,
            'total_models': len(self.models),
            'models': {}
        }

        current_time = time.time()

        for model_name, metadata in self.model_metadata.items():
            idle_time = current_time - metadata['last_accessed']
            stats['models'][model_name] = {
                'load_time': round(metadata['load_time'], 2),
                'access_count': metadata['access_count'],
                'idle_seconds': round(idle_time, 2),
                'idle_minutes': round(idle_time / 60, 2)
            }

        return stats

    def print_stats(self):
        """Print model statistics to console."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("MODEL MANAGER STATISTICS")
        print("=" * 60)
        print(f"Preload enabled: {stats['enabled']}")
        print(f"Total models loaded: {stats['total_models']}")

        if stats['models']:
            print("\nLoaded models:")
            for name, info in stats['models'].items():
                print(f"  {name}:")
                print(f"    Load time: {info['load_time']}s")
                print(f"    Access count: {info['access_count']}")
                print(f"    Idle time: {info['idle_minutes']:.1f} minutes")

        print("=" * 60 + "\n")


# Example loader functions for common models

def load_elevenlabs_client():
    """Example: Load ElevenLabs client."""
    from elevenlabs import ElevenLabs
    api_key = os.getenv("ELEVENLABS_API_KEY")
    return ElevenLabs(api_key=api_key)


def load_openai_client():
    """Example: Load OpenAI client."""
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


# Convenience function to get the singleton instance
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    return ModelManager()


def main():
    """Main entry point for model manager CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage AI model preloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model statistics
  python model_manager.py --stats

  # Clear all cached models
  python model_manager.py --clear

  # Unload unused models (idle > 1 hour)
  python model_manager.py --cleanup
        """
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show model statistics"
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached models"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Unload unused models"
    )

    parser.add_argument(
        "--max-idle",
        type=int,
        default=3600,
        help="Maximum idle time in seconds for cleanup (default: 3600)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get model manager
    manager = get_model_manager()

    # Execute commands
    if args.stats:
        manager.print_stats()

    elif args.clear:
        response = input("Are you sure you want to clear all cached models? (yes/no): ")
        if response.lower() == 'yes':
            manager.clear_all()
            print("All models cleared from memory")
        else:
            print("Operation cancelled")

    elif args.cleanup:
        count = manager.unload_unused(args.max_idle)
        print(f"Unloaded {count} unused models")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
