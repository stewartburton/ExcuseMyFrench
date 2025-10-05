# Content Generation Workflow

This document details the end‑to‑end pipeline used to create an *Excuse My French* episode. Each stage may be run manually or orchestrated through n8n for full automation.

## 1. Trend Retrieval

- Run `scripts/fetch_trends.py` to fetch top trending topics from Google Trends using PyTrends and insert them into a local SQLite database (`data/trends.db`).
- Optionally curate trending sounds or hashtags from TikTok or Instagram and store them manually.

## 2. Script Generation

- `scripts/generate_script.py` reads recent trends and uses an LLM (e.g., GPT‑4, Claude) to generate a dialogue between Butcher and Nutsy.  The prompt includes character descriptions, trending keywords and desired tone (humorous, sarcastic, wise).
- The output is a JSON file containing lines of dialogue and emotional cues for each character.

## 3. Audio Synthesis

- `scripts/generate_audio.py` iterates over the script JSON and calls ElevenLabs’ API to synthesize audio for each line.  Voice IDs for Butcher and Nutsy should be configured in `config/.env`.
- The script outputs individual MP3 files and records their durations.

## 4. Image Selection/Generation

- `scripts/select_images.py` selects existing images from the image library (`data/images/`) based on the emotional cue (e.g., happy, sad, excited).
- If no suitable image exists, `scripts/generate_images.py` generates a new frame using the DreamBooth model for Butcher and the base Stable Diffusion model for Nutsy.  Generated images are saved to `data/generated/` and added to the library.

## 5. Lip‑Sync and Animation

- `scripts/animate.py` uses SadTalker (or Wav2Lip) to animate each selected or generated image with its corresponding audio.  The result is a short video clip per sentence.
- Adjust parameters such as pose stabilization and expression intensity to ensure natural movement.

## 6. Optional Video Generation with Wan 2.2

- When you want a dynamic camera move or complex background, use ComfyUI with the Wan 2.2 workflow to transform a single frame and a text prompt into a 5–8 second video.  For example, you might prompt: “camera slowly zooms in on Butcher and Nutsy sitting at a picnic table while the sun sets.”
- This step is optional and best suited for special episodes or highlight scenes.

## 7. Assembly and Editing

- `scripts/assemble_video.py` concatenates the lip‑synced clips in order and aligns them on a timeline using their durations.  It also loads a music track generated via Mubert or Soundverse and mixes it at a low volume under the dialogue.
- Subtitles are burned into the video using FFmpeg.  Adjust audio levels, transitions and overall length to suit Instagram’s 60‑second limit.

## 8. Posting

- After manual review, upload the final video to Instagram using the Meta Graph API via `scripts/post_instagram.py` or schedule it through Creator Studio.
- Store engagement metrics (views, likes, comments) in `data/metrics.db` to analyse performance and inform future script prompts.