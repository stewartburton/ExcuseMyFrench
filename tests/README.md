# ExcuseMyFrench Test Suite

This directory contains the test suite for the ExcuseMyFrench project.

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov pytest-mock
```

### Run All Tests

```bash
# From project root
pytest tests/

# With coverage report
pytest tests/ --cov=scripts --cov-report=html

# Verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_post_instagram.py -v
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_post_instagram.py` - Tests for Instagram posting functionality
- `test_animate.py` - Tests for animation generation
- `test_prepare_training_data.py` - Tests for training data preparation

## Test Coverage

The tests cover:

1. **Instagram Posting (`test_post_instagram.py`)**
   - Caption sanitization (length, hashtags, encoding)
   - Video validation (format, size, codec)
   - Rate limiting
   - Token refresh
   - Duplicate detection

2. **Animation (`test_animate.py`)**
   - Path validation
   - Checkpoint/resume functionality
   - Timeline processing
   - Episode validation

3. **Data Preparation (`test_prepare_training_data.py`)**
   - Image validation
   - Caption generation
   - Directory structure

## Writing New Tests

When adding new features:

1. Create a test file named `test_<module_name>.py`
2. Use fixtures from `conftest.py` for common setup
3. Follow the existing test structure
4. Aim for high coverage of critical paths
5. Mock external dependencies (APIs, file I/O where appropriate)

## Fixtures

Available fixtures (see `conftest.py`):

- `temp_dir` - Provides a temporary directory for test files
- `mock_env_vars` - Sets up mock environment variables
- `sample_script_data` - Sample dialogue script data
- `sample_timeline_data` - Sample timeline for animation tests

## Continuous Integration

Tests are run automatically on:
- Pull requests
- Commits to main branch

See `.github/workflows/` for CI configuration.
