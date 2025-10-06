# Excuse My French - Pipeline Scripts

This directory contains the core pipeline scripts for generating Excuse My French videos. The scripts process content in the following order:

1. **fetch_trends.py** - Fetch trending topics from Google Trends
2. **generate_script.py** - Generate dialogue scripts using LLM
3. **generate_audio.py** - Generate voice audio using ElevenLabs
4. **select_images.py** - Select character images from library
5. **generate_images.py** - Generate missing images with Stable Diffusion
6. **assemble_video.py** - Assemble final video with ffmpeg

## Prerequisites

### System Requirements
- Python 3.10 or later
- FFmpeg installed and in PATH
- (Recommended) NVIDIA GPU with 12GB+ VRAM for image generation

### Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg:
- **Windows**: Download from https://ffmpeg.org/download.html
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt-get install ffmpeg`

3. Configure API keys in `config/.env`:
```bash
cp config/.env.example config/.env
# Edit config/.env and add your API keys
```

### Required API Keys
- **ElevenLabs API Key** (required for audio generation)
- **Anthropic API Key** OR **OpenAI API Key** (required for script generation)
- **Hugging Face Token** (optional, for downloading Stable Diffusion models)

## Pipeline Overview

```
Trending Topics → Script → Audio → Images → Video
     ↓              ↓        ↓        ↓        ↓
fetch_trends → generate_ → generate_ → select_ → assemble_
               script      audio      images    video
                                        ↓
                                   generate_
                                   images
                                   (if needed)
```

## Script Usage

### 1. fetch_trends.py

Fetch trending topics from Google Trends and store in SQLite database.

**Basic usage:**
```bash
# Fetch current US trends
python scripts/fetch_trends.py

# Fetch trends for specific region
python scripts/fetch_trends.py --region india

# Fetch interest data for specific keywords
python scripts/fetch_trends.py --keywords "AI" "robots" "technology"

# Show recent trends from database
python scripts/fetch_trends.py --show-recent --days 7 --limit 10
```

**Output:**
- Creates/updates `data/trends.db` with trending keywords

### 2. generate_script.py

Generate dialogue scripts using LLM (Anthropic Claude or OpenAI GPT).

**Basic usage:**
```bash
# Generate script from recent trends
python scripts/generate_script.py

# Generate script from specific topics
python scripts/generate_script.py --topics "artificial intelligence" "robots"

# Use specific LLM provider
python scripts/generate_script.py --provider anthropic

# Preview without saving
python scripts/generate_script.py --dry-run
```

**Output:**
- Creates `data/scripts/episode_YYYYMMDD_HHMMSS.json` with dialogue

**Script format:**
```json
{
  "timestamp": "20240101_120000",
  "characters": ["Butcher", "Nutsy"],
  "script": [
    {
      "character": "Butcher",
      "line": "Have you seen this AI nonsense?",
      "emotion": "sarcastic"
    },
    {
      "character": "Nutsy",
      "line": "AI is going to take over the world!",
      "emotion": "excited"
    }
  ]
}
```

### 3. generate_audio.py

Generate audio files using ElevenLabs TTS with character-specific voices.

**Basic usage:**
```bash
# Generate audio from script
python scripts/generate_audio.py data/scripts/episode_20240101_120000.json

# Custom output directory
python scripts/generate_audio.py script.json --output-dir data/audio/test

# Custom episode name
python scripts/generate_audio.py script.json --episode-name pilot
```

**Output:**
- Creates `data/audio/EPISODE_NAME/` directory containing:
  - `001_butcher_sarcastic.mp3`
  - `002_nutsy_excited.mp3`
  - `timeline.json` (timing information)

**Note:** Respects ElevenLabs rate limits (default: 3 requests/second)

### 4. select_images.py

Query the image library database to select character images for each line.

**Basic usage:**
```bash
# Select images for a script
python scripts/select_images.py data/scripts/episode_20240101.json --output selections.json

# Scan directory and add images to library
python scripts/select_images.py --scan data/butcher_images --character Butcher --emotion neutral

# Show library statistics
python scripts/select_images.py --stats

# Query for specific image
python scripts/select_images.py --query --character Butcher --emotion sarcastic
```

**Output:**
- Creates image selections JSON file
- Reports missing images that need to be generated

**First-time setup:**
```bash
# Add Butcher training images
python scripts/select_images.py --scan data/butcher_images --character Butcher --emotion neutral

# If you have Nutsy images
python scripts/select_images.py --scan data/nutsy_images --character Nutsy --emotion neutral
```

### 5. generate_images.py

Generate missing character images using Stable Diffusion.

**Basic usage:**
```bash
# Generate single image
python scripts/generate_images.py --character Butcher --emotion sarcastic

# Generate from missing images list
python scripts/generate_images.py --missing-json missing_images.json

# Use CPU instead of GPU
python scripts/generate_images.py --character Nutsy --emotion excited --cpu

# Control generation quality
python scripts/generate_images.py --character Butcher --emotion happy --steps 75 --guidance 8.0
```

**Output:**
- Creates `data/generated/CHARACTER/` directory with generated images
- Automatically adds images to library database

**Notes:**
- For Butcher: Uses DreamBooth model from `models/dreambooth_butcher` (train first!)
- For Nutsy: Uses base Stable Diffusion model
- Requires significant GPU memory (8GB+ VRAM recommended)
- Generation takes 30-60 seconds per image on GPU

### 6. assemble_video.py

Assemble final video with audio, images, and optional music/subtitles.

**Basic usage:**
```bash
# Assemble video from timeline and image selections
python scripts/assemble_video.py \
  --timeline data/audio/20240101_120000/timeline.json \
  --images selections.json

# Add background music
python scripts/assemble_video.py \
  --timeline data/audio/episode/timeline.json \
  --images selections.json \
  --music data/music/background.mp3 \
  --music-volume 0.15

# Skip subtitles
python scripts/assemble_video.py \
  --timeline timeline.json \
  --images selections.json \
  --no-subtitles
```

**Output:**
- Creates `data/final_videos/EPISODE_NAME.mp4`
- Also creates `EPISODE_NAME.srt` subtitle file

## Complete Workflow Example

Here's a complete end-to-end example:

```bash
# 1. Fetch trending topics
python scripts/fetch_trends.py
python scripts/fetch_trends.py --show-recent

# 2. Generate script from trends
python scripts/generate_script.py
# Output: data/scripts/episode_20240101_120000.json

# 3. Generate audio
python scripts/generate_audio.py data/scripts/episode_20240101_120000.json
# Output: data/audio/20240101_120000/

# 4. Select images (first time: scan your image library)
python scripts/select_images.py --scan data/butcher_images --character Butcher --emotion neutral
python scripts/select_images.py data/scripts/episode_20240101_120000.json --output selections.json
# Output: selections.json (with missing images list)

# 5. Generate missing images (if needed)
python scripts/generate_images.py --missing-json selections.json
# Then re-run image selection
python scripts/select_images.py data/scripts/episode_20240101_120000.json --output selections.json

# 6. Assemble final video
python scripts/assemble_video.py \
  --timeline data/audio/20240101_120000/timeline.json \
  --images selections.json \
  --music data/music/background.mp3
# Output: data/final_videos/20240101_120000.mp4
```

## Database Management

The pipeline uses three SQLite databases:

1. **trends.db** - Stores trending topics and keywords
2. **image_library.db** - Indexes character images by emotion
3. **metrics.db** - (Future) Tracks video performance metrics

To initialize or reset databases, see `init_databases.py` and `README_DATABASES.md`.

## Troubleshooting

### Common Issues

**"No valid API key found"**
- Check that `config/.env` has ANTHROPIC_API_KEY or OPENAI_API_KEY
- Verify keys start with `sk-ant-` (Anthropic) or `sk-` (OpenAI)

**"ElevenLabs API error"**
- Check ELEVENLABS_API_KEY in `config/.env`
- Verify you haven't exceeded rate limits
- Check voice IDs are correct for your account

**"FFmpeg not found"**
- Install FFmpeg and add to system PATH
- Test with: `ffmpeg -version`

**"CUDA out of memory"**
- Reduce image generation batch size
- Use CPU instead: `--cpu` flag
- Close other GPU applications

**"No suitable image found"**
- Scan your image directories first
- Or generate images with `generate_images.py`
- Check `select_images.py --stats` to see coverage

### Performance Tips

1. **Image Generation**
   - Use GPU for 10x faster generation
   - Lower `--steps` (30-40) for faster, lower quality
   - Higher `--steps` (75-100) for better quality

2. **Audio Generation**
   - Rate limiting prevents API errors
   - Adjust ELEVENLABS_RATE_LIMIT in .env if needed

3. **Video Assembly**
   - Keep SAVE_INTERMEDIATE_FILES=true for debugging
   - Set to false to save disk space

## Next Steps

### Animation Integration (Future)

The current pipeline uses static images. To add animation:

1. Install SadTalker or Wav2Lip
2. Create `animate.py` script
3. Insert between `generate_audio.py` and `assemble_video.py`
4. Use animated clips instead of static images

See `docs/WORKFLOW.md` for more details on animation setup.

### n8n Automation

To automate the entire pipeline:

1. Install n8n
2. Import workflow from `workflows/` directory
3. Set up schedule triggers
4. Configure error notifications

## Additional Resources

- **Main Documentation**: `../README.md`
- **Workflow Guide**: `../docs/WORKFLOW.md`
- **Image Generation**: `../docs/IMAGE_GENERATION.md`
- **Environment Config**: `../config/README.md`

## Support

For issues or questions:
- Check existing issues on GitHub
- Review documentation in `docs/`
- Check environment configuration in `config/.env.example`
