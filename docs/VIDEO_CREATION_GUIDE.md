# Complete Video Creation Guide

**Last Updated:** October 7, 2025

This guide walks you through the **complete end-to-end process** of creating a video with Butcher and Nutsy, from script generation to final video assembly with lip-sync animation.

---

## Table of Contents

1. [Prerequisites Checklist](#prerequisites-checklist)
2. [Step 1: Generate Script](#step-1-generate-script)
3. [Step 2: Generate Character Voices](#step-2-generate-character-voices)
4. [Step 3: Generate Character Images](#step-3-generate-character-images)
5. [Step 4: Select Images for Timeline](#step-4-select-images-for-timeline)
6. [Step 5: Add Lip-Sync Animation](#step-5-add-lip-sync-animation)
7. [Step 6: Assemble Final Video](#step-6-assemble-final-video)
8. [Step 7: Post to Social Media (Optional)](#step-7-post-to-social-media-optional)
9. [Complete Example Walkthrough](#complete-example-walkthrough)
10. [Quick Reference](#quick-reference)

---

## Prerequisites Checklist

Before you start, make sure you have:

### ✅ Required Software
- [X] Python 3.10+ installed
- [X] NVIDIA GPU with 12GB+ VRAM (RTX 3060, 4070, or better)
- [X] CUDA 12.1 installed
- [X] FFmpeg installed (verify: `ffmpeg -version`)
- [X] Virtual environment activated: `excusemyfrench\Scripts\activate`

### ✅ Required API Keys (in `config/.env`)
- [X] `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (for script generation)
- [X] `ELEVENLABS_API_KEY` (for voice synthesis)
- [X] `ELEVENLABS_VOICE_BUTCHER` (Butcher's voice ID)
- [X] `ELEVENLABS_VOICE_NUTSY` (Nutsy's voice ID)
- [X] `HF_TOKEN` (HuggingFace token for model downloads)

### ✅ Required Models Downloaded
- [X] **DreamBooth Butcher Model** - `models/dreambooth_butcher/` (trained model)
- [X] **Wav2Lip Models** - `models/wav2lip/` (366MB)
  - `s3fd.pth` (86MB)
  - `Wav2Lip-SD-GAN.pt` (140MB) or `Wav2Lip-SD-NOGAN.pt` (140MB)
- [X] **SadTalker V0.0.2** - `models/sadtalker/checkpoints/` (1.4GB)
  - `SadTalker_V0.0.2_256.safetensors` (692MB)
  - `SadTalker_V0.0.2_512.safetensors` (692MB)
- [X] **GFPGAN Weights** - `models/sadtalker/gfpgan/weights/` (610MB)
  - `alignment_WFLW_4HG.pth` (185MB)
  - `GFPGANv1.4.pth` (333MB)
  - `headpose_hopenet.pth` (92MB)

### ✅ Verify Setup
```bash
# Check environment
make check-env

# Or verify manually:
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
ffmpeg -version
python scripts/demo.py
```

---

## Step 1: Generate Script

The script defines the dialogue between Butcher and Nutsy, along with timing and emotions.

### Option A: Generate from Trending Topics (Recommended)

This creates a script based on current trending topics from Google Trends.

```bash
# 1. Fetch latest trending topics
make fetch-trends
# Or: python scripts/fetch_trends.py

# 2. Generate script from trends (last 3 days)
python scripts/generate_script.py --from-trends --days 3
```

**Output:** `data/scripts/episode_YYYYMMDD_HHMMSS.json`

### Option B: Generate with Custom Topic

```bash
python scripts/generate_script.py --topic "cryptocurrency crash" --format short
```

### Option C: Provide Your Own Script

Create a JSON file in `data/scripts/` with this structure:

```json
{
  "episode_id": "episode_20251007_120000",
  "title": "Butcher's Take on AI",
  "script": [
    {
      "character": "Butcher",
      "emotion": "sarcastic",
      "text": "Oh great, another AI that thinks it's smarter than me.",
      "duration": 3.5
    },
    {
      "character": "Nutsy",
      "emotion": "excited",
      "text": "But Butcher! It can generate videos in seconds!",
      "duration": 2.8
    }
  ],
  "metadata": {
    "topic": "AI video generation",
    "target_duration": 30,
    "platform": "instagram"
  }
}
```

### Understanding the Script Output

The generated script includes:
- **character**: "Butcher" or "Nutsy"
- **emotion**: "sarcastic", "excited", "happy", "grumpy", "confused", etc.
- **text**: The dialogue line
- **duration**: Estimated speaking time in seconds

**Script Location:** `data/scripts/episode_*.json`

---

## Step 2: Generate Character Voices

Use ElevenLabs API to generate audio for each dialogue line with distinct character voices.

### Generate Audio

```bash
# Generate audio from the script
python scripts/generate_audio.py data/scripts/episode_20251007_120000.json

# Or use the latest script automatically
python scripts/generate_audio.py data/scripts/episode_*.json
```

### What Happens:
1. Reads the script JSON
2. For each dialogue line:
   - Sends text to ElevenLabs API with appropriate voice ID
   - Downloads generated audio (MP3)
   - Saves to `data/audio/episode_*/character_lineXX.mp3`
3. Creates `timeline.json` with timing information

### Output Structure:
```
data/audio/episode_20251007_120000/
├── timeline.json          # Timing and sequence information
├── butcher_line01.mp3     # Butcher's first line
├── nutsy_line02.mp3       # Nutsy's response
├── butcher_line03.mp3     # Butcher's second line
└── ...
```

### Timeline JSON Format:
```json
{
  "episode_id": "episode_20251007_120000",
  "total_duration": 30.5,
  "audio_segments": [
    {
      "character": "Butcher",
      "emotion": "sarcastic",
      "file": "butcher_line01.mp3",
      "start_time": 0.0,
      "duration": 3.5,
      "text": "Oh great, another AI that thinks it's smarter than me."
    },
    {
      "character": "Nutsy",
      "emotion": "excited",
      "file": "nutsy_line02.mp3",
      "start_time": 3.5,
      "duration": 2.8,
      "text": "But Butcher! It can generate videos in seconds!"
    }
  ]
}
```

### Troubleshooting:
- **401 Unauthorized**: Check `ELEVENLABS_API_KEY` in `config/.env`
- **Voice not found**: Verify `ELEVENLABS_VOICE_BUTCHER` and `ELEVENLABS_VOICE_NUTSY` are correct
- **Rate limit exceeded**: Wait 60 seconds or upgrade ElevenLabs plan

---

## Step 3: Generate Character Images

Use the trained DreamBooth model to generate consistent character images matching the emotions in the script.

### Generate Images with DreamBooth

```bash
# Generate Butcher image with specific emotion
python scripts/generate_character_image.py \
  --character butcher \
  --emotion happy \
  --prompt "a photo of sks dog, happy expression, looking at camera" \
  --output data/images/butcher_happy_001.png

# Generate multiple emotion variants
python scripts/generate_character_image.py \
  --character butcher \
  --emotion sarcastic \
  --prompt "a photo of sks dog, sarcastic smirk, side-eye glance" \
  --output data/images/butcher_sarcastic_001.png

python scripts/generate_character_image.py \
  --character butcher \
  --emotion grumpy \
  --prompt "a photo of sks dog, grumpy expression, furrowed brow" \
  --output data/images/butcher_grumpy_001.png
```

### Character Image Guidelines:

**Butcher (French Bulldog):**
- Base prompt: `"a photo of sks dog, [emotion] expression"`
- Emotions: happy, sarcastic, grumpy, confused, excited, skeptical
- Additional details: "looking at camera", "studio lighting", "neutral background"

**Nutsy (Squirrel):**
- Base prompt: `"a photo of a hyperactive squirrel, [emotion] expression"`
- Emotions: excited, curious, energetic, mischievous, surprised
- Additional details: "bright eyes", "fluffy tail", "perched on branch"

### Advanced: Batch Generate Images

```bash
# Generate images for all emotions in the script
python scripts/batch_generate_images.py \
  --script data/scripts/episode_20251007_120000.json \
  --output-dir data/images/episode_20251007_120000/
```

### Image Output:
```
data/images/episode_20251007_120000/
├── butcher_sarcastic_001.png
├── nutsy_excited_001.png
├── butcher_grumpy_002.png
└── ...
```

### Troubleshooting:
- **CUDA out of memory**: Reduce image size or use `--fp16` flag
- **DreamBooth model not found**: Verify training completed and model exists in `models/dreambooth_butcher/`
- **Poor quality images**: Adjust prompts with more specific details, try different seeds with `--seed XXXXX`

---

## Step 4: Select Images for Timeline

Match generated images to the audio timeline based on character and emotion.

### Automatic Image Selection

```bash
# Select images automatically based on timeline emotions
python scripts/select_images.py data/scripts/episode_20251007_120000.json
```

### What Happens:
1. Reads `timeline.json` from audio directory
2. For each audio segment:
   - Finds matching character images
   - Selects image with closest emotion match
   - Falls back to neutral/default if exact match not found
3. Creates `image_selections.json` mapping audio to images

### Manual Image Selection (Optional)

Edit `data/image_selections.json` to manually choose specific images:

```json
{
  "episode_id": "episode_20251007_120000",
  "selections": [
    {
      "start_time": 0.0,
      "duration": 3.5,
      "character": "Butcher",
      "emotion": "sarcastic",
      "audio_file": "data/audio/episode_20251007_120000/butcher_line01.mp3",
      "image_file": "data/images/episode_20251007_120000/butcher_sarcastic_001.png"
    },
    {
      "start_time": 3.5,
      "duration": 2.8,
      "character": "Nutsy",
      "emotion": "excited",
      "audio_file": "data/audio/episode_20251007_120000/nutsy_line02.mp3",
      "image_file": "data/images/episode_20251007_120000/nutsy_excited_001.png"
    }
  ]
}
```

### Verify Selections:

```bash
# Check that all images exist
python scripts/verify_image_selections.py data/image_selections.json
```

---

## Step 5: Add Lip-Sync Animation

Animate the character images to match the audio using Wav2Lip or SadTalker.

### Option A: Wav2Lip (Fast, Good Quality)

Recommended for quick turnaround and good lip-sync accuracy.

```bash
# Animate with Wav2Lip GAN model (high quality)
python scripts/animate.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --images data/image_selections.json \
  --method wav2lip \
  --quality high \
  --output data/animated/episode_20251007_120000/

# Or use NOGAN model (faster, slightly lower quality)
python scripts/animate.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --images data/image_selections.json \
  --method wav2lip-nogan \
  --quality medium \
  --output data/animated/episode_20251007_120000/
```

**Processing Time:** ~30-60 seconds per segment on RTX 4070

### Option B: SadTalker (High Quality, Emotional)

Better for expressive animations with head movement and emotions.

```bash
# Animate with SadTalker (256x256 for speed)
python scripts/animate.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --images data/image_selections.json \
  --method sadtalker \
  --resolution 256 \
  --output data/animated/episode_20251007_120000/

# Or use 512x512 for higher quality
python scripts/animate.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --images data/image_selections.json \
  --method sadtalker \
  --resolution 512 \
  --output data/animated/episode_20251007_120000/
```

**Processing Time:** ~2-5 minutes per segment on RTX 4070

### Animation Parameters:

| Parameter | Wav2Lip | SadTalker | Description |
|-----------|---------|-----------|-------------|
| `--quality` | low, medium, high | N/A | Wav2Lip quality preset |
| `--resolution` | N/A | 256, 512 | SadTalker output resolution |
| `--face-enhance` | ✅ | ✅ | Apply GFPGAN face enhancement |
| `--batch-size` | 4, 8, 16 | 1, 2 | GPU batch size (higher = faster but more VRAM) |

### Output Structure:
```
data/animated/episode_20251007_120000/
├── segment_00_butcher.mp4     # Animated first line
├── segment_01_nutsy.mp4       # Animated second line
├── segment_02_butcher.mp4     # Animated third line
└── ...
```

### Troubleshooting:
- **CUDA out of memory**: Reduce `--batch-size` or use lower resolution
- **Face not detected**: Ensure images show clear frontal face, adjust with `--face-det-batch-size 1`
- **Poor lip-sync**: Try SadTalker instead of Wav2Lip, or regenerate image with clearer mouth area

---

## Step 6: Assemble Final Video

Combine all animated segments into a final video with transitions and effects.

### Basic Assembly

```bash
# Assemble video from animated segments
python scripts/assemble_video.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --animated data/animated/episode_20251007_120000/ \
  --output data/final_videos/episode_20251007_120000.mp4
```

### Advanced Assembly with Effects

```bash
# Add transitions, background music, and captions
python scripts/assemble_video.py \
  --timeline data/audio/episode_20251007_120000/timeline.json \
  --animated data/animated/episode_20251007_120000/ \
  --transitions fade \
  --background-music data/music/background_01.mp3 \
  --captions \
  --caption-style bottom \
  --output data/final_videos/episode_20251007_120000.mp4
```

### Assembly Parameters:

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--transitions` | none, fade, wipe, slide | Transition effect between segments |
| `--transition-duration` | 0.1 - 2.0 | Transition duration in seconds (default: 0.3) |
| `--background-music` | path/to/music.mp3 | Add background music (auto-ducked during dialogue) |
| `--music-volume` | 0.0 - 1.0 | Background music volume (default: 0.2) |
| `--captions` | flag | Add text captions for accessibility |
| `--caption-style` | bottom, top, center | Caption position |
| `--resolution` | 720p, 1080p | Output resolution (default: 1080p) |
| `--fps` | 24, 30, 60 | Output frame rate (default: 30) |
| `--format` | mp4, mov | Output format (default: mp4) |

### Output:
```
data/final_videos/episode_20251007_120000.mp4
```

### Verify Video:

```bash
# Check video properties
ffprobe -v error -show_entries format=duration,size,bit_rate -show_entries stream=width,height,codec_name,r_frame_rate data/final_videos/episode_20251007_120000.mp4

# Play video (Windows)
start data/final_videos/episode_20251007_120000.mp4

# Play video (Linux/Mac)
# ffplay data/final_videos/episode_20251007_120000.mp4
```

### Troubleshooting:
- **FFmpeg not found**: Run `ffmpeg -version` to verify installation
- **Audio/video out of sync**: Check timeline.json timing accuracy, reduce transition duration
- **Large file size**: Reduce resolution to 720p or lower bitrate with `--bitrate 2M`

---

## Step 7: Post to Social Media (Optional)

Automatically post your video to Instagram or other platforms.

### Instagram Posting

**Prerequisites:**
1. Configure Instagram API - see [docs/INSTAGRAM_SETUP.md](INSTAGRAM_SETUP.md)
2. Add to `config/.env`:
   ```bash
   META_ACCESS_TOKEN=your_token
   INSTAGRAM_USER_ID=your_user_id
   ```

### Test Posting (Dry-Run)

```bash
# Test posting without actually posting
python scripts/post_instagram.py \
  data/final_videos/episode_20251007_120000.mp4 \
  --dry-run \
  --caption "Butcher's hot take on AI! 🐕🤖 #ExcuseMyFrench #AI"
```

### Post for Real

```bash
# Post to Instagram
python scripts/post_instagram.py \
  data/final_videos/episode_20251007_120000.mp4 \
  --caption "Butcher's hot take on AI! 🐕🤖 #ExcuseMyFrench #AI" \
  --hashtags "AI,TechHumor,DogComedy,ButcherAndNutsy"
```

### Caption Best Practices:

- **Length**: 125 characters or less (appears fully in feed)
- **Hashtags**: 3-5 relevant hashtags (avoid spam)
- **Emojis**: 1-3 emojis for personality
- **Call-to-action**: "Follow for more!" or "Share if you agree!"

### Posting Parameters:

| Parameter | Description |
|-----------|-------------|
| `--caption` | Caption text (auto-generated if not provided) |
| `--hashtags` | Comma-separated hashtags |
| `--location-id` | Instagram location ID (optional) |
| `--schedule` | Post at specific time (YYYY-MM-DD HH:MM) |
| `--dry-run` | Test without actually posting |

### Troubleshooting:
- **401 Unauthorized**: Check `META_ACCESS_TOKEN` in `.env`, may need to refresh token
- **Video too long**: Instagram Reels max 90 seconds, reduce script or split into parts
- **Upload failed**: Check video meets Instagram specs (aspect ratio 9:16 or 1:1, max 100MB)

---

## Complete Example Walkthrough

Let's create a complete video from start to finish!

### Example Scenario: "Butcher Reacts to Bitcoin Crash"

```bash
# ========================================
# STEP 1: GENERATE SCRIPT
# ========================================
python scripts/generate_script.py \
  --topic "bitcoin crash 2025" \
  --format short \
  --duration 30

# Output: data/scripts/episode_20251007_143000.json

# ========================================
# STEP 2: GENERATE VOICES
# ========================================
python scripts/generate_audio.py \
  data/scripts/episode_20251007_143000.json

# Output: data/audio/episode_20251007_143000/
#   - timeline.json
#   - butcher_line01.mp3
#   - nutsy_line02.mp3
#   - ... (8-10 audio files)

# ========================================
# STEP 3: GENERATE IMAGES
# ========================================
# Generate Butcher images
python scripts/generate_character_image.py \
  --character butcher \
  --emotion sarcastic \
  --prompt "a photo of sks dog, sarcastic smirk, looking at camera" \
  --output data/images/butcher_sarcastic_001.png

python scripts/generate_character_image.py \
  --character butcher \
  --emotion grumpy \
  --prompt "a photo of sks dog, grumpy expression, furrowed brow" \
  --output data/images/butcher_grumpy_001.png

# Generate Nutsy images
python scripts/generate_character_image.py \
  --character nutsy \
  --emotion excited \
  --prompt "a photo of a hyperactive squirrel, excited expression, bright eyes" \
  --output data/images/nutsy_excited_001.png

python scripts/generate_character_image.py \
  --character nutsy \
  --emotion confused \
  --prompt "a photo of a hyperactive squirrel, confused expression, head tilted" \
  --output data/images/nutsy_confused_001.png

# ========================================
# STEP 4: SELECT IMAGES
# ========================================
python scripts/select_images.py \
  data/scripts/episode_20251007_143000.json

# Output: data/image_selections.json

# ========================================
# STEP 5: ANIMATE WITH LIP-SYNC
# ========================================
python scripts/animate.py \
  --timeline data/audio/episode_20251007_143000/timeline.json \
  --images data/image_selections.json \
  --method wav2lip \
  --quality high \
  --face-enhance \
  --output data/animated/episode_20251007_143000/

# Output: data/animated/episode_20251007_143000/
#   - segment_00_butcher.mp4
#   - segment_01_nutsy.mp4
#   - ... (8-10 animated segments)

# ========================================
# STEP 6: ASSEMBLE FINAL VIDEO
# ========================================
python scripts/assemble_video.py \
  --timeline data/audio/episode_20251007_143000/timeline.json \
  --animated data/animated/episode_20251007_143000/ \
  --transitions fade \
  --captions \
  --resolution 1080p \
  --output data/final_videos/butcher_bitcoin_crash.mp4

# Output: data/final_videos/butcher_bitcoin_crash.mp4

# ========================================
# STEP 7: POST TO INSTAGRAM (OPTIONAL)
# ========================================
python scripts/post_instagram.py \
  data/final_videos/butcher_bitcoin_crash.mp4 \
  --caption "Butcher reacts to the Bitcoin crash 😂💰 #ExcuseMyFrench #Bitcoin #Crypto" \
  --hashtags "Crypto,Bitcoin,DogComedy,TechHumor"

# Video posted to Instagram! 🎉
```

### Expected Timeline:
1. Script generation: ~30 seconds
2. Voice generation: ~1-2 minutes (8-10 lines)
3. Image generation: ~2-3 minutes (4 images × 30-45 seconds each)
4. Image selection: ~5 seconds
5. Animation: ~5-8 minutes (8-10 segments × 30-60 seconds each with Wav2Lip)
6. Video assembly: ~30-60 seconds
7. Instagram posting: ~15-30 seconds

**Total Time:** ~10-15 minutes from start to published video!

---

## Quick Reference

### One-Command Pipeline (Automated)

```bash
# Run the complete pipeline automatically
make run-pipeline
```

This executes all steps in sequence:
1. Fetch trending topics
2. Generate script from trends
3. Generate audio
4. Generate and select images
5. Animate with lip-sync
6. Assemble final video

**Output:** `data/final_videos/episode_*.mp4`

### Individual Commands

```bash
# 1. Fetch trends
make fetch-trends

# 2. Generate script
python scripts/generate_script.py --from-trends --days 3

# 3. Generate audio
python scripts/generate_audio.py data/scripts/episode_*.json

# 4. Generate images
python scripts/generate_character_image.py --character butcher --emotion sarcastic --prompt "a photo of sks dog, sarcastic expression"

# 5. Select images
python scripts/select_images.py data/scripts/episode_*.json

# 6. Animate
python scripts/animate.py --timeline data/audio/*/timeline.json --images data/image_selections.json --method wav2lip

# 7. Assemble video
python scripts/assemble_video.py --timeline data/audio/*/timeline.json --animated data/animated/*/

# 8. Post to Instagram
python scripts/post_instagram.py data/final_videos/episode_*.mp4 --caption "Check out Butcher's latest take! 🐕"
```

### File Locations Reference

| File Type | Location |
|-----------|----------|
| Scripts | `data/scripts/episode_*.json` |
| Audio | `data/audio/episode_*/` |
| Generated images | `data/images/` or `data/images/episode_*/` |
| Image selections | `data/image_selections.json` |
| Animated segments | `data/animated/episode_*/` |
| Final videos | `data/final_videos/` |
| Training images | `training/butcher/images/` |
| Models | `models/` |

### Common Makefile Commands

```bash
make setup              # Initial setup
make check-env          # Verify environment
make run-pipeline       # Complete automated pipeline
make fetch-trends       # Fetch trending topics
make train-butcher      # Train DreamBooth model
make test               # Run all tests
make test-comfyui       # Test ComfyUI integration
make test-instagram     # Test Instagram posting (dry-run)
make clean              # Clean temporary files
make backup-db          # Backup databases
make stats              # Show project statistics
make help               # Show all commands
```

### Character Voice IDs

**Butcher (Sarcastic French Bulldog):**
- Voice: Deep, gravelly, sarcastic tone
- ElevenLabs Voice ID: Set in `ELEVENLABS_VOICE_BUTCHER` env variable
- Emotions: sarcastic, grumpy, skeptical, deadpan

**Nutsy (Hyperactive Squirrel):**
- Voice: High-pitched, energetic, enthusiastic
- ElevenLabs Voice ID: Set in `ELEVENLABS_VOICE_NUTSY` env variable
- Emotions: excited, curious, mischievous, surprised

### Troubleshooting Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| CUDA out of memory | Reduce batch size or resolution |
| FFmpeg not found | `choco install ffmpeg -y` (Windows admin) |
| API key not found | Check `config/.env` file exists and has correct keys |
| Model download fails | Check `HF_TOKEN` in `.env`, accept license on HuggingFace |
| Poor lip-sync quality | Try SadTalker instead of Wav2Lip, or use `--quality high` |
| Video too large | Reduce resolution to 720p: `--resolution 720p` |
| Instagram upload fails | Check video is <90 seconds and <100MB |

### Performance Tips

1. **Use Wav2Lip for speed** (~30-60 seconds per segment vs 2-5 minutes for SadTalker)
2. **Generate images in advance** (create library of emotions before video generation)
3. **Parallel processing** (generate audio and images simultaneously in separate terminals)
4. **GPU utilization** (monitor with `nvidia-smi`, adjust batch sizes for 90%+ GPU usage)
5. **Cache everything** (reuse generated images across multiple videos)

---

## Next Steps

### Build Your Image Library
Create a collection of character images with various emotions:

```bash
# Create batch generation script
mkdir -p data/images/library/butcher
mkdir -p data/images/library/nutsy

# Generate 10 Butcher emotions
for emotion in happy sarcastic grumpy skeptical confused excited angry surprised smug deadpan; do
  python scripts/generate_character_image.py \
    --character butcher \
    --emotion $emotion \
    --prompt "a photo of sks dog, $emotion expression, studio lighting" \
    --output data/images/library/butcher/${emotion}_001.png
done
```

### Automate with n8n
Set up daily automated video generation:

1. Install n8n: `npm install -g n8n`
2. Import workflow: `n8n import:workflow --input=n8n/workflows/daily_video.json`
3. Configure schedule (e.g., daily at 9 AM)
4. Enable auto-posting to Instagram

### Experiment with Styles
Try different animation styles and effects:

- **Wan 2.2 video generation** - Create dynamic video backgrounds
- **ComfyUI workflows** - Advanced image generation with ControlNet
- **Custom transitions** - Add unique visual effects between segments
- **Background music** - Curate mood-appropriate soundtracks

---

## Support and Resources

- **Troubleshooting Guide**: [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Character Profiles**: [docs/CHARACTER_PROFILES.md](CHARACTER_PROFILES.md)
- **Instagram Setup**: [docs/INSTAGRAM_SETUP.md](INSTAGRAM_SETUP.md)
- **Model Downloads**: [docs/MODEL_DOWNLOADS.md](MODEL_DOWNLOADS.md)
- **Project Status**: [PROJECT_STATUS.md](../PROJECT_STATUS.md)
- **TODO List**: [TODO.md](../TODO.md)

---

**Happy creating! 🎬🐕🐿️**
