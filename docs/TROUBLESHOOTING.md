# Troubleshooting Guide

Common issues and solutions for Excuse My French.

---

## Installation & Setup Issues

### Python Version Errors

**Problem:** `SyntaxError` or incompatible Python version

**Solution:**
```bash
# Check Python version
python --version  # Should be 3.10+

# If too old, install Python 3.10+
# On Ubuntu/Debian:
sudo apt install python3.10

# On macOS with Homebrew:
brew install python@3.10

# Create environment with specific version
conda create -n excusemyfrench python=3.10
```

### pip install fails

**Problem:** Package installation errors

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install with verbose output to see errors
pip install -r requirements.txt -v

# If specific package fails, install it separately
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### CUDA / GPU Issues

**Problem:** `CUDA out of memory` or `No CUDA-capable device detected`

**Solution:**
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Reduce memory usage in .env
ANIMATION_BATCH_SIZE=2
ANIMATION_QUALITY=low
```

---

## API & Authentication Issues

### OpenAI API Errors

**Problem:** `openai.error.AuthenticationError`

**Solution:**
```bash
# Verify API key in .env
cat config/.env | grep OPENAI

# Test API key
python -c "from openai import OpenAI; client = OpenAI(); print('✓ Valid')"

# Common issues:
# 1. Missing 'sk-' prefix
# 2. Extra spaces or quotes
# 3. Expired key
```

**Problem:** `Rate limit exceeded`

**Solution:**
```python
# Reduce request frequency
# In generate_script.py, add delays between requests
import time
time.sleep(2)  # Wait 2 seconds between requests
```

### ElevenLabs API Errors

**Problem:** `elevenlabs.api.error.UnauthenticatedRateLimitError`

**Solution:**
```bash
# Check API key
python -c "from elevenlabs import ElevenLabs; client = ElevenLabs(api_key='your_key'); print('✓ Valid')"

# Free tier limits:
# - 10,000 characters/month
# - 3 custom voices

# Upgrade plan or reduce usage
```

**Problem:** Voice IDs not found

**Solution:**
```bash
# List available voices
python -c "from elevenlabs import ElevenLabs; client = ElevenLabs(); voices = client.voices.get_all(); print([v.voice_id for v in voices.voices])"

# Update .env with correct voice IDs
```

### Instagram API Errors

**Problem:** `InstagramAPI.Error: Invalid access token`

**Solution:**
- See `docs/INSTAGRAM_SETUP.md` for complete setup guide
- Access tokens expire after 60 days
- Refresh token: https://developers.facebook.com/docs/instagram-basic-display-api/guides/long-lived-access-tokens

---

## Database Issues

### Database Locked

**Problem:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Close all scripts using the database
pkill -f "python scripts/"

# Or restart and reinitialize
make backup-db
rm data/*.db
python scripts/init_databases.py
```

### Schema Mismatch

**Problem:** `sqlite3.OperationalError: no such table`

**Solution:**
```bash
# Reinitialize databases
python scripts/init_databases.py --reset

# Restore data from backup if needed
```

### Data Corruption

**Problem:** Database file is corrupt

**Solution:**
```bash
# Try to recover
sqlite3 data/content.db ".dump" | sqlite3 data/content_recovered.db

# Or restore from backup
cp backups/backup_YYYYMMDD_HHMMSS/content.db data/

# Last resort: reset
mv data/content.db data/content.db.backup
python scripts/init_databases.py
```

---

## Pipeline & Generation Issues

### No Trending Topics Found

**Problem:** `fetch_trends.py` returns empty results

**Solution:**
```bash
# Try with more days and keywords
python scripts/fetch_trends.py --days 30 --limit 50

# Try different timeframes
python scripts/fetch_trends.py --timeframe today-1-m

# Check Google Trends manually to verify connectivity
```

### Script Generation Fails

**Problem:** `generate_script.py` produces no output or errors

**Solution:**
```python
# Check if trends exist in database
python -c "import sqlite3; conn = sqlite3.connect('data/content.db'); \
cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM trending_topics'); \
print(f'Trends: {cursor.fetchone()[0]}')"

# Generate with custom topics if no trends
python scripts/generate_script.py --topics "French bulldogs,squirrels,park adventures"

# Enable debug logging
python scripts/generate_script.py --verbose
```

### Audio Generation Fails

**Problem:** `generate_audio.py` fails or produces no sound

**Solution:**
```bash
# Check ElevenLabs quota
# Log into ElevenLabs dashboard to check remaining characters

# Test with single line
python scripts/generate_audio.py test_script.json --verbose

# Verify ffmpeg is installed
ffmpeg -version

# Check output file
ffplay data/audio/episode/001_butcher_happy.mp3
```

### Video Assembly Fails

**Problem:** `assemble_video.py` crashes or produces corrupt video

**Solution:**
```bash
# Verify FFmpeg installation and codecs
ffmpeg -codecs | grep h264

# Check input files exist
ls -lh data/audio/*/timeline.json
ls -lh data/image_selections.json

# Test with minimal settings
python scripts/assemble_video.py --timeline <path> --images <path> --verbose

# Common FFmpeg issues:
# 1. Missing codec: install ffmpeg with h264 support
# 2. Invalid paths: use absolute paths
# 3. Permissions: check file ownership
```

---

## Model & Animation Issues

### ComfyUI Won't Start

**Problem:** ComfyUI server fails to start

**Solution:**
```bash
# Check ComfyUI installation
ls $COMFYUI_PATH/main.py

# Manual start with debug output
cd $COMFYUI_PATH
python main.py --verbose

# Common issues:
# 1. Missing models: download required checkpoints
# 2. Port conflict: change port in config
# 3. CUDA issues: verify GPU drivers
```

### SadTalker/Wav2Lip Not Working

**Problem:** Animation fails or produces static images

**Solution:**
```bash
# Verify model checkpoints
ls -lh models/sadtalker/
ls -lh models/wav2lip/

# Download missing models
# SadTalker: https://github.com/OpenTalker/SadTalker#-installation
# Wav2Lip: https://github.com/Rudrabha/Wav2Lip#getting-the-weights

# Test with sample files
python scripts/animate.py \
  --image test.jpg \
  --audio test.mp3 \
  --method sadtalker \
  --verbose
```

### DreamBooth Training Fails

**Problem:** `train_dreambooth.py` crashes or OOM

**Solution:**
```bash
# Reduce batch size in config
# Edit training/config/butcher_config.yaml
train_batch_size: 1
gradient_accumulation_steps: 4

# Use gradient checkpointing
enable_gradient_checkpointing: true

# Lower resolution
resolution: 512  # instead of 1024

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

## Performance Issues

### Slow Generation

**Problem:** Pipeline takes too long

**Solution:**
```bash
# Use lower quality settings
ANIMATION_QUALITY=low
VIDEO_QUALITY=medium

# Reduce resolution
VIDEO_WIDTH=720
VIDEO_HEIGHT=1280

# Skip expensive steps
# Disable animation, use static images
python scripts/assemble_video.py --no-animation

# Parallelize when possible
# Generate multiple scripts in parallel (with rate limits)
```

### High Memory Usage

**Problem:** System runs out of RAM/VRAM

**Solution:**
```bash
# Close other applications
# Reduce batch sizes
ANIMATION_BATCH_SIZE=1

# Use CPU fallback for some operations
TORCH_DEVICE=cpu  # in .env

# Monitor memory
watch -n 1 free -h
```

---

## Network & Connectivity Issues

### API Timeouts

**Problem:** Requests to APIs timeout

**Solution:**
```python
# Increase timeout in scripts
# Edit generate_script.py or generate_audio.py
import openai
openai.api_timeout = 60  # 60 seconds

# Check network connectivity
ping api.openai.com
ping api.elevenlabs.io

# Use retry logic (already implemented in updated scripts)
```

### Proxy Issues

**Problem:** Behind corporate firewall/proxy

**Solution:**
```bash
# Set proxy in environment
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# Or add to .env
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080
```

---

## File & Permission Issues

### Permission Denied

**Problem:** `PermissionError` when writing files

**Solution:**
```bash
# Fix directory permissions
chmod -R 755 data/
chmod -R 755 models/

# Check ownership
ls -la data/

# Run with proper user
sudo chown -R $USER:$USER data/ models/
```

### Disk Space Issues

**Problem:** No space left on device

**Solution:**
```bash
# Check disk usage
df -h
du -sh data/*

# Clean temporary files
make clean

# Remove old generated content
rm -rf data/animated/old_episodes/
rm -rf data/final_videos/old_*

# Set up automatic cleanup (in .env)
SAVE_INTERMEDIATE_FILES=false
```

---

## Docker Issues

### Docker Build Fails

**Problem:** `docker build` errors

**Solution:**
```bash
# Build with verbose output
docker build --progress=plain -t excusemyfrench:latest .

# Check Docker version
docker --version  # Need 20.10+

# Clear build cache
docker builder prune

# Build without cache
docker build --no-cache -t excusemyfrench:latest .
```

### Container Won't Start

**Problem:** Docker container exits immediately

**Solution:**
```bash
# Check logs
docker logs excusemyfrench

# Run interactively
docker-compose run --rm excusemyfrench /bin/bash

# Common issues:
# 1. Missing .env file
# 2. GPU not accessible: install nvidia-docker2
# 3. Volume mount issues: check paths
```

### GPU Not Available in Docker

**Problem:** CUDA not available inside container

**Solution:**
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Getting More Help

### Enable Debug Logging

```bash
# Add to all scripts
--verbose

# Or set in .env
LOG_LEVEL=DEBUG
```

### Collect Diagnostic Info

```bash
# System info
python --version
pip list
nvidia-smi
ffmpeg -version

# Project status
make stats
python scripts/init_databases.py --check

# Package versions
pip list | grep -E "openai|elevenlabs|torch|ffmpeg"
```

### Report an Issue

When reporting issues, include:
1. Error message (full stack trace)
2. Python version: `python --version`
3. GPU info: `nvidia-smi`
4. OS: `uname -a` (Linux/Mac) or `ver` (Windows)
5. Steps to reproduce
6. Relevant config (redact API keys!)

**GitHub Issues:** https://github.com/stewartburton/ExcuseMyFrench/issues

---

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `ModuleNotFoundError: No module named 'X'` | Missing dependency | `pip install X` |
| `FileNotFoundError: config/.env` | Missing config | Copy `.env.example` to `.env` |
| `CUDA out of memory` | GPU memory exhausted | Reduce batch size, close other GPU apps |
| `Rate limit exceeded` | Too many API requests | Wait, or upgrade API plan |
| `Database is locked` | Multiple processes accessing DB | Close other scripts |
| `FFmpeg not found` | FFmpeg not installed | `sudo apt install ffmpeg` |
| `Invalid access token` | Expired or wrong token | Regenerate API token |
| `No such table: X` | Database not initialized | Run `python scripts/init_databases.py` |

---

## Prevention Tips

1. **Always backup before major changes:** `make backup-db`
2. **Test with dry-run first:** `--dry-run` flag available on most scripts
3. **Monitor API quotas:** Check dashboards regularly
4. **Keep dependencies updated:** `pip install -U -r requirements.txt`
5. **Use version control:** Commit working states
6. **Read error messages:** They usually tell you exactly what's wrong
7. **Check logs:** `*.log` files in project root

---

## Still Stuck?

1. Check the docs: `docs/` directory
2. Search existing issues: GitHub Issues
3. Ask for help: Open a new issue with diagnostic info
4. Join community: [Discord/Forum if available]

Remember: Most issues have simple solutions. Take a deep breath, read the error message carefully, and try the solutions above! 🎬
