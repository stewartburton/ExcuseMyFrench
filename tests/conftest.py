"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for tests."""
    env_vars = {
        "META_ACCESS_TOKEN": "test_token_123",
        "INSTAGRAM_USER_ID": "test_user_123",
        "INSTAGRAM_POST_ENABLED": "false",
        "ANTHROPIC_API_KEY": "sk-ant-test123",
        "ANIMATION_QUALITY": "medium",
        "VIDEO_WIDTH": "1080",
        "VIDEO_HEIGHT": "1920",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sample_script_data():
    """Provide sample script data for testing."""
    return [
        {
            "character": "Butcher",
            "line": "Did you hear about the latest AI trend?",
            "emotion": "sarcastic"
        },
        {
            "character": "Nutsy",
            "line": "Oh my gosh, YES! It's so exciting!",
            "emotion": "excited"
        },
        {
            "character": "Butcher",
            "line": "You don't even know what it is, do you?",
            "emotion": "neutral"
        }
    ]


@pytest.fixture
def sample_timeline_data():
    """Provide sample timeline data for testing."""
    return {
        "episode": "test_episode_001",
        "total_duration": 15.0,
        "lines": [
            {
                "index": 1,
                "character": "Butcher",
                "line": "Test line 1",
                "emotion": "sarcastic",
                "audio_file": "data/audio/test/001_butcher.mp3",
                "start_time": 0.0,
                "duration": 5.0
            },
            {
                "index": 2,
                "character": "Nutsy",
                "line": "Test line 2",
                "emotion": "excited",
                "audio_file": "data/audio/test/002_nutsy.mp3",
                "start_time": 5.0,
                "duration": 5.0
            }
        ]
    }
