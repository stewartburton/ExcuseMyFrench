# Testing Guide - Excuse My French

This document provides comprehensive testing procedures for the ExcuseMyFrench project, including unit tests, integration tests, and end-to-end testing workflows.

---

## Table of Contents

- [Testing Philosophy](#testing-philosophy)
- [Test Environment Setup](#test-environment-setup)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Performance Testing](#performance-testing)
- [Manual Testing Checklists](#manual-testing-checklists)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting Test Failures](#troubleshooting-test-failures)

---

## Testing Philosophy

**Key Principles:**

1. **Test Early, Test Often** - Catch issues before they reach production
2. **Automate Where Possible** - Reduce manual testing burden
3. **Test Real Scenarios** - Use realistic data and workflows
4. **Isolate Failures** - Make it easy to identify root causes
5. **Performance Matters** - Track and improve execution times

---

## Test Environment Setup

### Prerequisites

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-mock

# Verify environment
python scripts/validate_env.py --verbose

# Initialize test databases
python scripts/init_databases.py
```

### Environment Variables for Testing

Create a test-specific environment file:

```bash
# Copy and modify for testing
cp config/.env config/.env.test

# Set test-specific values
TEST_MODE=true
LOG_LEVEL=DEBUG
STRUCTURED_LOGGING=false

# Use test API keys (with rate limits)
# Or use mock services
USE_MOCK_APIS=true
```

### Test Data

```bash
# Create test data directory
mkdir -p data/test/

# Copy sample data
cp -r data/scripts/sample_episode.json data/test/
cp -r training/butcher/images/sample_*.jpg data/test/
```

---

## Unit Testing

### Running Unit Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test file
pytest tests/test_animate.py -v

# Run with coverage report
pytest tests/ --cov=scripts --cov-report=html

# Run fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"
```

### Writing Unit Tests

**Example test structure:**

```python
# tests/test_my_script.py
import pytest
from scripts.my_script import MyClass

class TestMyClass:
    """Tests for MyClass functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.instance = MyClass()

    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        result = self.instance.my_method("input")
        assert result == "expected_output"

    def test_error_handling(self):
        """Test that errors are handled correctly."""
        with pytest.raises(ValueError):
            self.instance.my_method(None)

    def test_with_mock(self, mocker):
        """Test using mocked dependencies."""
        mock_api = mocker.patch('scripts.my_script.api_call')
        mock_api.return_value = "mocked_response"

        result = self.instance.method_using_api()
        assert result == "processed_mocked_response"
        mock_api.assert_called_once()
```

### Current Test Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| `scripts/animate.py` | 85% | ✅ Tests written |
| `scripts/post_instagram.py` | 80% | ✅ Tests written |
| `scripts/generate_audio.py` | 65% | 🟡 Needs more tests |
| `scripts/assemble_video.py` | 60% | 🟡 Needs more tests |
| `scripts/select_images.py` | 70% | 🟡 Needs more tests |
| Other scripts | 30% | 🔴 Needs tests |

**Goal:** 80%+ coverage for all core scripts

---

## Integration Testing

### Component Integration Tests

Integration tests verify that multiple components work together correctly.

#### Database Integration

```bash
# Test database operations
pytest tests/integration/test_database.py -v
```

**Test checklist:**
- ✅ Database initialization
- ✅ Concurrent access handling
- ✅ Transaction rollback
- ✅ Migration compatibility
- ✅ Batch operations

#### API Integration

```bash
# Test external API integrations (requires API keys)
pytest tests/integration/test_apis.py -v --api-tests
```

**Test checklist:**
- ✅ OpenAI/Anthropic API calls
- ✅ ElevenLabs voice generation
- ✅ Instagram Graph API
- ✅ Rate limiting behavior
- ✅ Token refresh mechanism
- ✅ Error recovery

#### Pipeline Integration

```bash
# Test pipeline component interactions
pytest tests/integration/test_pipeline.py -v -m slow
```

**Test checklist:**
- ✅ Script generation → Audio generation
- ✅ Audio generation → Image selection
- ✅ Image selection → Video assembly
- ✅ Data format compatibility
- ✅ Error propagation

---

## End-to-End Testing

### Full Pipeline Test

Complete workflow from trending topics to final video.

#### Prerequisites

1. **Configure API keys** in `config/.env`
2. **Ensure models are downloaded** (if testing with models)
3. **Free disk space** (at least 5GB for test outputs)

#### Test Procedure

```bash
# Run automated end-to-end test
pytest tests/e2e/test_full_pipeline.py -v -s

# Or run manual end-to-end workflow
bash tests/e2e/run_full_workflow.sh
```

#### Manual End-to-End Test (Step-by-Step)

##### Step 1: Environment Validation (5 minutes)

```bash
# Verify environment is ready
python scripts/validate_env.py --verbose

# Expected output: All checks pass
```

**Success Criteria:**
- ✅ All required API keys present
- ✅ Required system dependencies installed
- ✅ Python packages available
- ✅ Directory structure correct
- ✅ Database files accessible

##### Step 2: Fetch Trending Topics (2 minutes)

```bash
# Fetch recent trends
python scripts/fetch_trends.py --days 7 --limit 10

# Verify trends were saved
sqlite3 data/trends.db "SELECT COUNT(*) FROM trends;"
# Should return > 0
```

**Success Criteria:**
- ✅ At least 5 trends fetched
- ✅ Trends saved to database
- ✅ No API errors
- ✅ Valid trend data (name, score, timestamp)

##### Step 3: Generate Script (3 minutes)

```bash
# Generate script from trends
python scripts/generate_script.py \
  --from-trends \
  --days 3 \
  --output data/test/test_episode.json

# Verify script structure
cat data/test/test_episode.json | jq '.lines | length'
# Should return > 0
```

**Success Criteria:**
- ✅ Script file created
- ✅ Valid JSON structure
- ✅ At least 5 dialogue lines
- ✅ Both characters present
- ✅ Emotions specified for each line
- ✅ Trending topics referenced

**Validation:**
```python
import json
with open('data/test/test_episode.json') as f:
    script = json.load(f)
    assert 'episode' in script
    assert 'metadata' in script
    assert 'lines' in script
    assert len(script['lines']) >= 5
    characters = {line['character'] for line in script['lines']}
    assert 'Butcher' in characters
    assert 'Nutsy' in characters
```

##### Step 4: Generate Audio (5 minutes)

```bash
# Generate audio for all lines
python scripts/generate_audio.py data/test/test_episode.json

# Verify audio files created
ls -lh data/audio/test_episode/*.mp3
# Should show multiple MP3 files

# Verify timeline created
cat data/audio/test_episode/timeline.json | jq '.'
```

**Success Criteria:**
- ✅ Audio file for each dialogue line
- ✅ Timeline JSON created
- ✅ Audio files are valid (playable)
- ✅ Correct duration recorded
- ✅ Character voices sound different
- ✅ No API rate limit errors

**Validation:**
```bash
# Check audio file validity
for file in data/audio/test_episode/*.mp3; do
  ffprobe -v error -show_format "$file" || echo "Invalid: $file"
done

# Check timeline integrity
python -c "
import json
with open('data/audio/test_episode/timeline.json') as f:
    timeline = json.load(f)
    assert 'episode' in timeline
    assert 'lines' in timeline
    assert timeline['total_duration'] > 0
    print(f'Total duration: {timeline[\"total_duration\"]}s')
"
```

##### Step 5: Select Images (2 minutes)

```bash
# Select images for each character/emotion
python scripts/select_images.py data/test/test_episode.json

# Verify selections created
cat data/image_selections.json | jq '.selections | length'
# Should match number of lines in script
```

**Success Criteria:**
- ✅ Image selection file created
- ✅ Selection for each dialogue line
- ✅ Image paths are valid
- ✅ Images exist on disk
- ✅ Correct character/emotion mapping

**Validation:**
```python
import json
from pathlib import Path

with open('data/image_selections.json') as f:
    selections = json.load(f)
    assert 'selections' in selections
    for sel in selections['selections']:
        assert Path(sel['image_path']).exists()
        print(f"Line {sel['line_index']}: {sel['image_path']}")
```

##### Step 6: Assemble Video (3 minutes)

```bash
# Assemble final video
python scripts/assemble_video.py \
  --timeline data/audio/test_episode/timeline.json \
  --images data/image_selections.json \
  --output data/final_videos/test_output.mp4

# Verify video created
ls -lh data/final_videos/test_output.mp4

# Check video properties
ffprobe -v error -show_format -show_streams data/final_videos/test_output.mp4
```

**Success Criteria:**
- ✅ Video file created
- ✅ Video is playable
- ✅ Correct duration (matches timeline)
- ✅ Audio track present
- ✅ Video resolution correct (1080x1920 for Instagram Reels)
- ✅ No encoding errors

**Validation:**
```bash
# Extract video info
python -c "
import json
import subprocess

result = subprocess.run([
    'ffprobe', '-v', 'error',
    '-show_format', '-show_streams',
    '-of', 'json',
    'data/final_videos/test_output.mp4'
], capture_output=True, text=True)

info = json.loads(result.stdout)
duration = float(info['format']['duration'])
print(f'Video duration: {duration}s')

# Check has video and audio streams
streams = {s['codec_type'] for s in info['streams']}
assert 'video' in streams
assert 'audio' in streams
print('Video validation: PASSED')
"
```

##### Step 7: Visual/Audio Quality Check (5 minutes)

**Manual review:**
1. Open `data/final_videos/test_output.mp4` in a video player
2. Verify:
   - ✅ Audio is clear and audible
   - ✅ Correct character voices
   - ✅ Images match characters
   - ✅ Smooth transitions
   - ✅ No stuttering or artifacts
   - ✅ Proper timing (image changes match dialogue)
   - ✅ Aspect ratio correct for Instagram (9:16)

##### Step 8: Metadata Validation (2 minutes)

```bash
# Check metrics were recorded
sqlite3 data/metrics.db "SELECT * FROM generation_metrics ORDER BY timestamp DESC LIMIT 1;"

# Check image library updated
sqlite3 data/image_library.db "SELECT character, emotion, usage_count FROM image_library ORDER BY usage_count DESC LIMIT 5;"
```

**Success Criteria:**
- ✅ Metrics recorded for generation
- ✅ Duration times tracked
- ✅ Image usage counts incremented
- ✅ Episode marked as generated

#### Cleanup

```bash
# Optional: Clean up test files
rm -rf data/test/
rm data/final_videos/test_output.mp4
rm data/image_selections.json
```

### End-to-End Test Results Template

```markdown
## E2E Test Report - [Date]

**Tester:** [Name]
**Environment:** [Dev/Staging/Prod]
**Duration:** [Total time]

### Results

| Step | Status | Duration | Notes |
|------|--------|----------|-------|
| 1. Environment Validation | ✅ PASS | 30s | All checks passed |
| 2. Fetch Trends | ✅ PASS | 45s | 10 trends fetched |
| 3. Generate Script | ✅ PASS | 2m 15s | 8 lines generated |
| 4. Generate Audio | ✅ PASS | 4m 30s | All voices correct |
| 5. Select Images | ✅ PASS | 45s | All paths valid |
| 6. Assemble Video | ✅ PASS | 2m 10s | 30s video created |
| 7. Quality Check | ✅ PASS | 5m | Audio/video quality good |
| 8. Metadata | ✅ PASS | 15s | All records updated |

**Overall Result:** ✅ PASS
**Total Time:** 16 minutes 10 seconds

### Issues Found
- None

### Recommendations
- None
```

---

## Performance Testing

### Performance Benchmarks

Track execution time for each pipeline stage.

```bash
# Run performance benchmark
python tests/performance/benchmark_pipeline.py

# Sample output:
# ====================================
# Pipeline Performance Benchmark
# ====================================
# Fetch Trends:        0.8s  (10 trends)
# Generate Script:   125.3s  (8 lines, GPT-4)
# Generate Audio:    245.7s  (8 files, ElevenLabs)
#   - With parallelization: 82.1s  (3x speedup)
# Select Images:       2.1s  (8 selections)
#   - With batch queries: 0.2s  (10x speedup)
# Assemble Video:    128.4s  (30s video, 1080p)
# ====================================
# Total Pipeline:    584.5s  (9m 44s)
# ====================================
```

### Load Testing

Test system under load with multiple concurrent operations.

```bash
# Generate 100 videos (simulated load)
python tests/performance/load_test.py --videos 100 --workers 4

# Monitor system resources
# - CPU usage
# - Memory usage
# - GPU memory (if applicable)
# - Disk I/O
# - API rate limits
```

**Expected results:**
- Memory usage stays under 8GB
- No memory leaks after 100 iterations
- Checkpoint/resume works correctly
- API rate limiting prevents errors
- Database handles concurrent access

---

## Manual Testing Checklists

### Pre-Release Checklist

Before deploying to production, verify:

#### Environment
- [ ] All API keys configured and valid
- [ ] Database files initialized
- [ ] Required directories exist
- [ ] Disk space sufficient (>50GB free)
- [ ] GPU drivers up to date (if using)

#### Core Pipeline
- [ ] Can fetch trending topics
- [ ] Can generate scripts from trends
- [ ] Can generate audio for both characters
- [ ] Can select/generate images
- [ ] Can assemble final video
- [ ] Video plays correctly
- [ ] Audio quality is good

#### Advanced Features
- [ ] DreamBooth model trained (if applicable)
- [ ] Animation working (if applicable)
- [ ] ComfyUI integration (if applicable)
- [ ] Instagram posting (dry-run successful)

#### Security & Validation
- [ ] Input validation working
- [ ] Path traversal protection active
- [ ] Rate limiting functional
- [ ] Error handling graceful
- [ ] Logging comprehensive

#### Documentation
- [ ] README up to date
- [ ] API keys documented
- [ ] Common issues documented
- [ ] Setup guide accurate

### Instagram Posting Checklist

Before posting to Instagram:

#### Pre-Post Verification
- [ ] Video file exists and is valid
- [ ] Video duration < 90 seconds (Reels limit)
- [ ] Video resolution 1080x1920 (9:16 aspect ratio)
- [ ] File size < 100MB
- [ ] Audio track present and audible
- [ ] Caption generated
- [ ] Caption length < 2200 characters
- [ ] Hashtag count <= 30
- [ ] No inappropriate content

#### Instagram API
- [ ] API credentials valid
- [ ] Token not expired (or auto-refresh works)
- [ ] Rate limit headroom available
- [ ] Dry-run successful

#### Post-Post Verification
- [ ] Post ID returned
- [ ] Post visible on Instagram
- [ ] Video plays correctly
- [ ] Caption formatted correctly
- [ ] Metrics recorded in database

---

## CI/CD Integration

### GitHub Actions Workflow

Example workflow for automated testing:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/ -v --cov=scripts --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
```

### Pre-Commit Hooks

Set up pre-commit hooks to run tests locally:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml <<EOF
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/, -v, -m, "not slow"]
EOF

# Install hooks
pre-commit install
```

---

## Troubleshooting Test Failures

### Common Test Failures

#### "API key not found"

**Cause:** Missing or invalid API keys in test environment

**Fix:**
```bash
# Verify API keys
cat config/.env | grep API_KEY

# Or use mock APIs for testing
export USE_MOCK_APIS=true
pytest tests/
```

#### "Database is locked"

**Cause:** Concurrent access to SQLite database

**Fix:**
```bash
# Close other scripts/tests
# Or use separate test database
export TEST_DB_PATH=data/test_databases/
pytest tests/
```

#### "CUDA out of memory" (GPU tests)

**Cause:** Insufficient GPU memory

**Fix:**
```bash
# Reduce batch size for tests
export TEST_BATCH_SIZE=1
pytest tests/

# Or skip GPU tests
pytest tests/ -m "not gpu"
```

#### "Test timeout"

**Cause:** API calls taking too long

**Fix:**
```bash
# Increase timeout
pytest tests/ --timeout=300

# Or use mocked APIs
pytest tests/ --mock-apis
```

### Debug Mode

Run tests with verbose output:

```bash
# Maximum verbosity
pytest tests/ -vv -s --log-cli-level=DEBUG

# Show print statements
pytest tests/ -s

# Drop into debugger on failure
pytest tests/ --pdb
```

### Test Data Issues

If tests fail due to bad test data:

```bash
# Reset test data
rm -rf data/test/
python tests/setup_test_data.py

# Re-run failed tests
pytest tests/ --lf  # --last-failed
```

---

## Best Practices

### Writing Good Tests

1. **Test One Thing** - Each test should verify a single behavior
2. **Use Descriptive Names** - Test names should explain what they test
3. **Arrange-Act-Assert** - Structure tests clearly (setup, execute, verify)
4. **Isolate Tests** - Tests should not depend on each other
5. **Use Fixtures** - Share setup code via pytest fixtures
6. **Mock External Services** - Don't rely on external APIs in unit tests
7. **Test Edge Cases** - Include boundary conditions and error cases
8. **Keep Tests Fast** - Mark slow tests appropriately

### Test Maintenance

1. **Run Tests Regularly** - Before each commit/push
2. **Fix Failures Immediately** - Don't let broken tests accumulate
3. **Update Tests with Code** - Keep tests in sync with changes
4. **Review Coverage** - Aim for 80%+ coverage of core functionality
5. **Refactor Tests** - Apply DRY principle to test code too

---

## DreamBooth Training Testing

### Testing DreamBooth Model Training

The DreamBooth training script includes checkpoint/resume functionality and requires GPU testing.

#### Prerequisites

```bash
# Ensure GPU is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Verify HuggingFace token is set
cat config/.env | grep HF_TOKEN

# Check training data exists
ls -l training/butcher/images/
# Should show 15-25 training images
```

#### Test 1: Basic Training Start (5 minutes)

Test that training can start and save the first checkpoint.

```bash
# Start training
python scripts/train_dreambooth.py \
  --config training/config/butcher_config.yaml

# Monitor GPU usage in another terminal
nvidia-smi -l 1

# Expected: Training starts, GPU utilization increases to 90%+
```

**Success Criteria:**
- ✅ Models load from HuggingFace successfully
- ✅ GPU detected and utilized
- ✅ Training progress bar appears
- ✅ Class images generated (if first run)
- ✅ Training loss decreases over steps
- ✅ No CUDA out of memory errors

**After ~12 minutes (step 100):**
- ✅ Checkpoint saved to `models/dreambooth_butcher/checkpoint-100/`
- ✅ Checkpoint contains: model.safetensors, optimizer.bin, scheduler.bin

#### Test 2: Checkpoint Resume (2 minutes)

Test that training can resume from a saved checkpoint.

```bash
# Cancel training with Ctrl+C after checkpoint-100 is saved

# Resume from latest checkpoint
python scripts/train_dreambooth.py \
  --config training/config/butcher_config.yaml \
  --resume

# Expected output should show:
# - "Resuming from checkpoint: models\dreambooth_butcher\checkpoint-100"
# - "Resuming from step 100"
# - Progress bar starts at 100/800 (not 0/800)
```

**Success Criteria:**
- ✅ Latest checkpoint detected automatically
- ✅ All model states loaded successfully
- ✅ Training resumes from correct step (100, not 0)
- ✅ Progress bar shows correct initial position
- ✅ Loss values are consistent with where training left off
- ✅ No re-generation of class images

#### Test 3: Validation Generation (3 minutes)

Test that validation images are generated at checkpoints.

```bash
# Let training continue to step 200
# Validation runs automatically at checkpoint steps

# After step 200, verify validation images exist
ls -l models/dreambooth_butcher/validation-200/
# Should show generated images

# View validation images to check quality
# Images should show the trained character
```

**Success Criteria:**
- ✅ Validation runs without errors
- ✅ Validation images generated in correct directory
- ✅ Images show character consistent with training data
- ✅ No dtype mismatch errors
- ✅ Training continues after validation

#### Test 4: Training Completion (1-2 hours)

Test full training cycle to completion.

```bash
# Let training run to completion (800 steps)
# ~1.5-2 hours on RTX 4070

# Monitor progress periodically
# Check for checkpoints at: 100, 200, 300, 400, 500, 600, 700

# After completion, verify final model
ls -l models/dreambooth_butcher/
# Should show: model files, model_info.json, all checkpoints
```

**Success Criteria:**
- ✅ Training completes all 800 steps
- ✅ Final model saved to output directory
- ✅ All checkpoints saved (every 100 steps)
- ✅ Validation images at each checkpoint
- ✅ model_info.json contains training metadata
- ✅ Training loss shows steady decrease
- ✅ No memory leaks (GPU memory stable)

#### Test 5: Model Inference (2 minutes)

Test generating images with the trained model.

```bash
# Generate test image with trained model
python scripts/generate_character_image.py \
  --character butcher \
  --emotion happy \
  --prompt "a photo of sks dog, happy expression, professional photography" \
  --model models/dreambooth_butcher/

# Verify image generated
ls -l data/images/generated/
```

**Success Criteria:**
- ✅ Image generated successfully
- ✅ Character appearance matches training images
- ✅ Emotion is correctly expressed
- ✅ Image quality is good (no artifacts)
- ✅ Consistent character appearance across multiple generations

### DreamBooth Troubleshooting

#### "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size in config
# Edit training/config/butcher_config.yaml:
train_batch_size: 1  # Already minimum
gradient_accumulation_steps: 2  # Increase this instead

# Or use smaller resolution
resolution: 512  # Instead of 768
```

#### "Connection timeout" downloading models

**Solution:**
```bash
# Pre-download base model
huggingface-cli login
huggingface-cli download stabilityai/stable-diffusion-2-1

# Then run training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
```

#### Loss not decreasing

**Solution:**
```bash
# Check learning rate (may be too high/low)
# Edit config:
learning_rate: 5e-6  # Standard value
learning_rate: 2e-6  # Try lower if loss unstable

# Check training data quality
# - Ensure 15-25 high-quality images
# - Verify images are diverse (angles, expressions)
# - Check images are properly cropped/centered
```

#### Validation fails with dtype error

**Solution:**
```bash
# This is fixed in the latest code
# If still occurring, update the script:
git pull origin main

# Ensure text_encoder and vae are passed to pipeline
# See scripts/train_dreambooth.py:690-694
```

---

## GPU Testing

### GPU Environment Verification

#### Test GPU Availability

```bash
# Check CUDA is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# Expected: CUDA available: True

# Check GPU details
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"
# Expected: GPU: NVIDIA GeForce RTX 4070 (or your GPU)

# Check CUDA version
python -c "import torch; print('CUDA version:', torch.version.cuda)"
# Expected: CUDA version: 12.1 (or your version)

# Check GPU memory
nvidia-smi --query-gpu=memory.total --format=csv
# Expected: 12282 MiB (for RTX 4070)
```

#### Test GPU Utilization

Monitor GPU during different operations:

```bash
# Terminal 1: Start GPU monitoring
nvidia-smi -l 1

# Terminal 2: Run GPU-intensive task
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

# Verify in Terminal 1:
# - GPU utilization reaches 90-100%
# - GPU memory usage increases to 8-10GB
# - Temperature stays under 85°C
# - Power usage near TDP limit
```

### GPU Memory Testing

#### Test Memory Limits

```bash
# Test maximum batch size before OOM
# Start with batch_size=1 and increase

# Edit training config temporarily
python -c "
import yaml
config = yaml.safe_load(open('training/config/butcher_config.yaml'))
config['train_batch_size'] = 2  # Increase from 1
yaml.dump(config, open('training/config/butcher_config_test.yaml', 'w'))
"

# Try training with larger batch
python scripts/train_dreambooth.py --config training/config/butcher_config_test.yaml

# If OOM occurs, you've found the limit
# Rollback to previous batch size
```

#### Test Memory Cleanup

```bash
# Verify GPU memory is freed after training
nvidia-smi --query-gpu=memory.used --format=csv
# Before training: ~1000 MiB (baseline)

# Start and stop training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
# Ctrl+C after a few steps

# Check memory again
nvidia-smi --query-gpu=memory.used --format=csv
# After training: ~1000 MiB (should return to baseline)

# If memory not freed, there's a leak
```

### GPU Performance Benchmarks

Expected performance on different GPUs:

| GPU Model | VRAM | Training Speed | Full Training Time |
|-----------|------|----------------|-------------------|
| RTX 4090 | 24GB | ~3-4s/step | 40-55 minutes |
| RTX 4070 Ti | 12GB | ~7-8s/step | 90-110 minutes |
| RTX 4070 | 12GB | ~7-8s/step | 90-110 minutes |
| RTX 3080 | 10GB | ~9-10s/step | 120-135 minutes |
| RTX 3060 | 12GB | ~12-15s/step | 160-200 minutes |

*For 800 training steps with batch_size=1, resolution=512*

---

## Checkpoint/Resume Testing

### Testing Checkpoint Functionality

The checkpoint/resume system should handle interruptions gracefully.

#### Test 1: Automatic Checkpoint Saving

```bash
# Start training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

# Monitor checkpoint directory
watch -n 10 ls -lh models/dreambooth_butcher/

# Verify checkpoints appear at steps: 100, 200, 300...
```

**Success Criteria:**
- ✅ Checkpoint directories created at correct intervals
- ✅ Each checkpoint contains all required files
- ✅ Checkpoint sizes are reasonable (~2-3GB each)
- ✅ Training continues without interruption

#### Test 2: Manual Interruption

```bash
# Start training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

# Wait for step ~150 (between checkpoints)
# Press Ctrl+C to interrupt

# Resume training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml --resume

# Verify resumes from step 100 (last checkpoint)
# Verify steps 100-150 are repeated (expected behavior)
```

**Success Criteria:**
- ✅ Graceful shutdown on Ctrl+C
- ✅ Resume finds correct checkpoint (100, not 150)
- ✅ Training continues normally
- ✅ Final model is identical to uninterrupted training

#### Test 3: Multiple Resume Cycles

```bash
# Resume, interrupt, resume multiple times
# This tests state persistence

# Cycle 1: Train 0→100, interrupt, resume
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
# Interrupt after step 100
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml --resume

# Cycle 2: Train 100→200, interrupt, resume
# Interrupt after step 200
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml --resume

# Cycle 3: Complete training
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml --resume
```

**Success Criteria:**
- ✅ Each resume loads correct checkpoint
- ✅ Loss values remain consistent
- ✅ Learning rate schedule preserved
- ✅ Optimizer state maintained
- ✅ Random seed state preserved
- ✅ Final model quality unaffected

#### Test 4: Checkpoint File Integrity

```bash
# Verify checkpoint file structure
ls -lR models/dreambooth_butcher/checkpoint-100/

# Should contain:
# - model.safetensors (model weights)
# - optimizer.bin (optimizer state)
# - scheduler.bin (LR scheduler state)
# - sampler.bin (data sampler state)
# - scaler.pt (gradient scaler state)
# - random_states_0.pkl (RNG state)

# Verify files are not corrupted
python -c "
import torch
import safetensors

# Test loading checkpoint files
ckpt = torch.load('models/dreambooth_butcher/checkpoint-100/optimizer.bin')
print(f'Optimizer checkpoint: {len(ckpt)} items')

# Test safetensors
from safetensors.torch import load_file
model = load_file('models/dreambooth_butcher/checkpoint-100/model.safetensors')
print(f'Model checkpoint: {len(model)} tensors')
"
```

**Success Criteria:**
- ✅ All required files present
- ✅ Files are not corrupted
- ✅ Files can be loaded successfully
- ✅ Checkpoint size matches expected (~2-3GB total)

---

## Resources

- **pytest Documentation:** https://docs.pytest.org/
- **pytest-cov Plugin:** https://pytest-cov.readthedocs.io/
- **Test-Driven Development:** https://testdriven.io/
- **PyTorch Testing:** https://pytorch.org/docs/stable/testing.html
- **HuggingFace Diffusers Testing:** https://huggingface.co/docs/diffusers/testing
- **Project Issue Tracker:** https://github.com/stewartburton/ExcuseMyFrench/issues

---

**Last Updated:** October 7, 2025
