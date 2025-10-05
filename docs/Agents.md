# Agents CLI Guide

This document describes how to interact with agents (e.g., OpenAI Codex or Claude CLI) to generate and post content for **Excuse My French**.  Agents automate tasks such as script generation, trending‑topic retrieval, image selection, audio synthesis, video assembly and posting.

## Overview

Agents are designed to run from the command line using Python scripts in the `scripts/` directory.  They communicate with external APIs (LLMs, ElevenLabs, AI music services) and local tools (SadTalker, Wan 2.2 via ComfyUI) to produce complete video episodes.

## Setup

1. **Activate your environment**:
   ```bash
   conda activate excusemyfrench
   ```
2. **Configure API keys** for your chosen LLM provider (OpenAI or Claude), ElevenLabs and optional AI music services in `config/.env`.
3. **Download necessary models** (Wan 2.2, DreamBooth checkpoint) and verify that ComfyUI is installed.

## Commands

The main entry point for agents is `scripts/agent.py`.  Use the `--mode` flag to choose between interactive and single‑command modes.

### generate_script
Fetches the latest trending topics and prompts the LLM to generate a short, humorous dialogue between Butcher and Nutsy.  You can specify a topic or let the script choose randomly.

**Usage:**
```bash
python scripts/agent.py generate_script --topic "braai"
```

### generate_audio
Reads a script JSON file and calls ElevenLabs to synthesize voices for each line.  Requires voice IDs configured in `config/.env`.

**Usage:**
```bash
python scripts/agent.py generate_audio --input scripts/output/dialogue.json --output data/audio
```

### generate_images
Selects appropriate images based on the emotional cues in the script.  If no suitable photo exists, it invokes the DreamBooth model or Stable Diffusion to create new images.

**Usage:**
```bash
python scripts/agent.py generate_images --input scripts/output/dialogue.json --output data/frames
```

### assemble_video
Uses SadTalker to animate the selected images with the synthesized audio and optionally uses Wan 2.2 for dynamic scenes.  Then stitches the clips together, overlays music and adds subtitles.

**Usage:**
```bash
python scripts/agent.py assemble_video --images data/frames --audio data/audio --music data/music/quippy_loop.mp3 --output data/final_video.mp4
```

### post
Uploads the final video to Instagram using Meta’s Graph API or schedules it via Creator Studio.  Stores engagement metrics in a local database.

**Usage:**
```bash
python scripts/agent.py post --video data/final_video.mp4 --caption "New episode: Butcher and Nutsy talk load shedding!"
```

## Examples

1. **Generate a Braai Day episode**:
   ```bash
   python scripts/agent.py generate_script --topic braai
   python scripts/agent.py generate_images
   python scripts/agent.py generate_audio
   python scripts/agent.py assemble_video --music data/music/braai_groove.mp3
   python scripts/agent.py post --caption "Happy Braai Day!"
   ```

2. **Create an episode using the most recent trending topic**:
   ```bash
   python scripts/agent.py generate_script
   ...
   ```

## Best Practices

* **Review generated content** before posting to ensure quality and compliance with platform guidelines.
* **Keep your API keys secure** by storing them in environment variables or a `.env` file that is not committed to the repository.
* **Iterate on prompts** to refine character voices and comedic timing.
* **Monitor performance** of your reels and adapt trending topics and prompts accordingly.
