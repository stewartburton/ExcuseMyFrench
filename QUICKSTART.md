# Quick Start Guide

Get **Excuse My French** up and running in 15 minutes!

## Prerequisites

Before you begin, make sure you have:

- ✅ Python 3.10 or later
- ✅ NVIDIA GPU with 12GB+ VRAM (RTX 3060, 4070, or better)
- ✅ Git installed
- ✅ 50GB+ free disk space

## Step 1: Clone and Install (5 minutes)

```bash
# Clone the repository
git clone https://github.com/stewartburton/ExcuseMyFrench.git
cd ExcuseMyFrench

# Quick setup with Make
make setup

# Or manual setup:
pip install -r requirements.txt
python scripts/init_databases.py
```

## Step 2: Configure API Keys (3 minutes)

```bash
# Copy the example environment file
cp config/.env.example config/.env

# Edit config/.env and add your API keys
```

**Required API Keys:**

1. **OpenAI API Key** (for script generation)
   - Get it from: https://platform.openai.com/api-keys
   - Add to `.env`: `OPENAI_API_KEY=sk-...`

2. **ElevenLabs API Key** (for voice generation)
   - Get it from: https://elevenlabs.io/
   - Add to `.env`: `ELEVENLABS_API_KEY=...`

3. **Instagram Access Token** (optional, for posting)
   - See `docs/INSTAGRAM_SETUP.md` for detailed instructions
   - Can skip for now and test locally

**Configure Voice IDs:**

1. Go to ElevenLabs and create/clone voices for Butcher and Nutsy
2. Use the voice descriptions in `docs/CHARACTER_PROFILES.md`
3. Add voice IDs to `.env`:
   ```
   ELEVENLABS_VOICE_BUTCHER=your_butcher_voice_id
   ELEVENLABS_VOICE_NUTSY=your_nutsy_voice_id
   ```

## Step 3: Verify Setup (2 minutes)

```bash
# Check that all API keys are configured
make check-env

# Test the database
python -c "import sqlite3; print('✓ Database OK' if sqlite3.connect('data/content.db') else '✗ Database failed')"
```

## Step 4: Generate Your First Video (5 minutes)

### Option A: Run Full Pipeline

```bash
# Run the complete pipeline
make run-pipeline
```

This will:
1. Fetch trending topics (30 seconds)
2. Generate a script (1 minute)
3. Generate audio for each line (2-3 minutes)
4. Select/generate character images (varies)
5. Assemble the final video (1 minute)

### Option B: Step-by-Step

```bash
# 1. Fetch trending topics
make fetch-trends

# 2. Generate script from trends
python scripts/generate_script.py --from-trends --days 3

# 3. Generate audio
python scripts/generate_audio.py data/scripts/episode_*.json

# 4. Select images (or generate if needed)
python scripts/select_images.py data/scripts/episode_*.json

# 5. Assemble video
python scripts/assemble_video.py \
  --timeline data/audio/*/timeline.json \
  --images data/image_selections.json
```

## Step 5: View Your Video

```bash
# Videos are saved in:
ls -lh data/final_videos/

# Open the video with your default player
```

---

## Quick Commands Reference

```bash
# Common operations
make help                 # Show all available commands
make run-pipeline         # Run complete pipeline
make fetch-trends         # Get latest trending topics
make stats                # Show project statistics
make clean                # Clean temporary files

# Testing
make test-comfyui         # Test image generation
make test-instagram       # Test Instagram posting (dry-run)

# Backup
make backup-db            # Backup all databases
```

---

## Next Steps

### 1. Customize Characters

Edit character voices and personalities:
- See `docs/CHARACTER_PROFILES.md` for voice settings
- Train custom image model: `make train-butcher`
- Test voices with sample paragraphs

### 2. Set Up Advanced Features

**Image Generation:**
```bash
# Install ComfyUI for advanced image generation
make setup-comfyui

# Generate custom character images
python scripts/generate_character_image.py --character butcher --emotion happy
```

**Animation:**
```bash
# Download SadTalker models
# Follow instructions in config/.env for SADTALKER_CHECKPOINT_PATH

# Animate images with lip-sync
python scripts/animate.py \
  --image data/images/butcher_001.jpg \
  --audio data/audio/episode/001.mp3
```

**Instagram Posting:**
```bash
# Configure Instagram API (see docs/INSTAGRAM_SETUP.md)
# Test posting
python scripts/post_instagram.py data/final_videos/episode_*.mp4 --dry-run

# Post for real
python scripts/post_instagram.py data/final_videos/episode_*.mp4
```

### 3. Automate with n8n

```bash
# Install n8n (workflow automation)
npm install -g n8n

# Import workflows
n8n import:workflow --input=n8n/workflows/main_pipeline.json

# Set up schedule (see docs for details)
```

---

## Troubleshooting

### "No module named 'elevenlabs'"
```bash
pip install elevenlabs
```

### "CUDA out of memory"
- Reduce batch size in `.env`: `ANIMATION_BATCH_SIZE=4`
- Use lower quality: `ANIMATION_QUALITY=medium`
- Close other GPU applications

### "API key not found"
```bash
# Verify .env file exists and has correct format
cat config/.env | grep API_KEY
```

### "No trending topics found"
```bash
# Try with more days
python scripts/fetch_trends.py --days 14 --limit 20
```

### "Database is locked"
```bash
# Close any other scripts accessing the database
# Or backup and reinitialize:
make backup-db
python scripts/init_databases.py --reset
```

---

## Common Workflows

### Daily Video Generation

```bash
# Automated daily workflow
0 9 * * * cd /path/to/ExcuseMyFrench && make run-pipeline
```

### Manual Content Creation

```bash
# 1. Write custom script
vim data/scripts/my_episode.json

# 2. Generate audio
python scripts/generate_audio.py data/scripts/my_episode.json

# 3. Select images manually
python scripts/select_images.py data/scripts/my_episode.json

# 4. Assemble
python scripts/assemble_video.py --timeline data/audio/*/timeline.json --images data/image_selections.json
```

### Batch Processing

```bash
# Generate multiple scripts
for i in {1..5}; do
  python scripts/generate_script.py --from-trends
  sleep 60
done

# Process all scripts
for script in data/scripts/episode_*.json; do
  make run-pipeline SCRIPT=$script
done
```

---

## Getting Help

- 📖 **Full Documentation:** See `docs/` directory
- 💬 **Issues:** https://github.com/stewartburton/ExcuseMyFrench/issues
- 📧 **Contact:** [Your contact info]

## What's Next?

1. ✅ Generated your first video
2. ⬜ Fine-tune character voices
3. ⬜ Train custom image models
4. ⬜ Set up automated posting
5. ⬜ Configure n8n workflows
6. ⬜ Customize video styles and music

Happy creating! 🎬🐕🐿️
