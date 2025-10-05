# Excuse My French

**Excuse My French** is an AI‑powered project that automatically generates short‑form videos starring **Butcher**, a sarcastic French bulldog, and **Nutsy**, his hyperactive squirrel sidekick.  The goal is to produce high‑quality, humorous content for platforms like Instagram and TikTok while keeping the workflow cost‑effective and largely self‑hosted.

## Features

- **Trending topic integration** – automatically pulls trending keywords from Google Trends and social networks to keep jokes topical and timely.
- **Script generation** – uses a large language model (LLM) to write conversations between Butcher and Nutsy with personality‑rich dialogue.
- **Audio synthesis** – generates distinct voices for each character via the ElevenLabs API.
- **Image generation** – trains a custom DreamBooth/LoRA model to produce new images of Butcher and uses Stable Diffusion for Nutsy when no photo is available.
- **Lip‑sync & animation** – animates still photos to match dialogue using SadTalker or Wav2Lip.
- **Video generation** – leverages Wan 2.2 to turn static images and prompts into cinematic video clips when dynamic motion is desired.
- **Music generation & mixing** – uses AI services such as Mubert or Soundverse to generate background music and mixes it with dialogue using FFmpeg.
- **Orchestration** – uses n8n to schedule tasks, call models and combine outputs into a complete reel.

## Getting Started

### Prerequisites

* Python 3.10 or later
* Conda (recommended for environment management)
* An NVIDIA GPU with at least 12 GB VRAM (RTX 4070 or better)
* ElevenLabs API key
* Wan 2.2 models downloaded (5B or 14B) and placed under `models/wan2.2/`
* [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installed and updated

### Installation

Clone this repository and set up your environment:

```bash
git clone https://github.com/stewartburton/ExcuseMyFrench.git
cd ExcuseMyFrench

# Create and activate a new environment
conda create -n excusemyfrench python=3.10
conda activate excusemyfrench

# Install Python dependencies (add more as needed)
pip install -r requirements.txt
```

Copy or symlink your Wan 2.2 model files into `models/wan2.2/`.  Place Butcher training photos in `data/butcher_images/`.  Follow the guidance in `docs/IMAGE_GENERATION.md` to train your DreamBooth model.  Configure API keys in `config/.env`.

### Usage

See `docs/WORKFLOW.md` for a detailed description of the automated pipeline and how to run it manually or via n8n.  There is also a CLI guide in `docs/Agents.md` for interacting with your LLM agents.

## Repository Structure

```
ExcuseMyFrench/
  ├── docs/               # Additional documentation (workflow, image generation, agents)
  ├── models/             # Model files (not included in version control)
  ├── scripts/            # Utility scripts (pytrends, audio synthesis, video assembly)
  ├── data/               # Training data and generated assets
  ├── config/             # Configuration files and environment variables
  ├── flowchart.png       # Visual overview of the pipeline
  └── README.md           # This file
  ```
  
  ## Contributing
  
  Contributions are welcome!  Please read `CONTRIBUTING.md` for guidelines on how to contribute to this project.
  
  ## License
  
  This project is licensed under the [Apache 2.0 License](LICENSE).
  
