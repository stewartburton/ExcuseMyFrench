# Excuse My French

> **Status:** 🟡 Core Complete, Testing Phase | [View Project Status](PROJECT_STATUS.md) | [View TODO List](TODO.md)

**Excuse My French** is an AI‑powered project that automatically generates short‑form videos starring **Butcher**, a sarcastic French bulldog, and **Nutsy**, his hyperactive squirrel sidekick. The goal is to produce high‑quality, humorous content for platforms like Instagram and TikTok while keeping the workflow cost‑effective and largely self‑hosted.

## Quick Start

Get up and running in 15 minutes! See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

```bash
# Clone and setup
git clone https://github.com/stewartburton/ExcuseMyFrench.git
cd ExcuseMyFrench
make setup

# Configure API keys (see QUICKSTART.md)
cp config/.env.example config/.env
# Edit config/.env with your API keys

# Run the pipeline
make run-pipeline
```

## Features

### Core Pipeline (✅ Complete)

- **Trending topic integration** – automatically pulls trending keywords from Google Trends and social networks to keep jokes topical and timely
- **Script generation** – uses a large language model (LLM) to write conversations between Butcher and Nutsy with personality‑rich dialogue
- **Audio synthesis** – generates distinct voices for each character via the ElevenLabs API with parallel processing (3x faster)
- **Video assembly** – combines audio, images, and effects using FFmpeg to create polished final videos
- **Database management** – tracks trends, metrics, and image library with optimized batch queries (10x faster)
- **Security & validation** – comprehensive input validation, path traversal protection, and rate limiting

### Advanced Features (🟡 In Progress)

- **Image generation** – trains a custom DreamBooth/LoRA model to produce consistent images of Butcher (23 training images ready)
- **Lip‑sync & animation** – animates still photos to match dialogue using SadTalker or Wav2Lip
- **ComfyUI integration** – generates character images with custom Stable Diffusion workflows and caching (150x faster)
- **Instagram posting** – automated posting with token refresh, rate limiting, and caption optimization
- **Performance optimizations** – model preloading (800x faster), parallel processing, and checkpoint/resume functionality
- **Orchestration** – uses n8n to schedule tasks and automate the complete workflow

## Getting Started

### Prerequisites

**Required:**
- Python 3.10 or later
- NVIDIA GPU with 12GB+ VRAM (RTX 3060, 4070, or better)
- FFmpeg (for video processing)
- 50GB+ free disk space

**API Keys (Required):**
- OpenAI or Anthropic API key (for script generation)
- ElevenLabs API key (for voice synthesis)

**Optional (for advanced features):**
- Wan 2.2 models (for video generation)
- ComfyUI (for advanced image generation)
- Instagram API access (for automated posting)
- Animation models (SadTalker or Wav2Lip) - [Download Guide](MODELS_QUICKSTART.md)

### Installation

```bash
# Clone the repository
git clone https://github.com/stewartburton/ExcuseMyFrench.git
cd ExcuseMyFrench

# Create virtual environment
python -m venv excusemyfrench
source excusemyfrench/bin/activate  # On Windows: excusemyfrench\Scripts\activate

# Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# Windows (requires admin PowerShell):
choco install ffmpeg -y
# Linux (Ubuntu/Debian):
# sudo apt install ffmpeg
# macOS:
# brew install ffmpeg

# Initialize databases
python scripts/init_databases.py

# Verify GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Verify FFmpeg is installed
ffmpeg -version
```

### Configuration

1. **Copy the environment template:**
   ```bash
   cp config/.env.example config/.env
   ```

2. **Add your API keys to `config/.env`:**
   ```bash
   # Required
   OPENAI_API_KEY=sk-...                    # Or ANTHROPIC_API_KEY
   ELEVENLABS_API_KEY=your_key_here
   ELEVENLABS_VOICE_BUTCHER=voice_id
   ELEVENLABS_VOICE_NUTSY=voice_id

   # HuggingFace (for model downloads)
   HF_TOKEN=hf_...                          # Get from https://huggingface.co/settings/tokens

   # Optional (for Instagram posting)
   META_ACCESS_TOKEN=your_token
   INSTAGRAM_USER_ID=your_user_id
   ```

3. **Verify your setup:**
   ```bash
   make check-env
   ```

### Usage

**Generate a video:**
```bash
make run-pipeline
```

This will:
1. Fetch trending topics
2. Generate a script
3. Synthesize audio for each character
4. Select/generate images
5. Assemble the final video

**View your video:**
```bash
ls -lh data/final_videos/
```

**Step-by-step workflow:**
```bash
# 1. Fetch trends
make fetch-trends

# 2. Generate script
python scripts/generate_script.py --from-trends --days 3

# 3. Generate audio
python scripts/generate_audio.py data/scripts/episode_*.json

# 4. Select images
python scripts/select_images.py data/scripts/episode_*.json

# 5. Assemble video
python scripts/assemble_video.py \
  --timeline data/audio/*/timeline.json \
  --images data/image_selections.json
```

## Repository Structure

```
ExcuseMyFrench/
├── config/                    # Configuration files and API keys
│   ├── .env                   # Environment variables (API keys)
│   └── README.md              # Configuration guide
├── data/                      # Generated content and databases
│   ├── scripts/               # Generated episode scripts
│   ├── audio/                 # Generated audio files
│   ├── images/                # Character images
│   ├── animated/              # Animated videos (with lip-sync)
│   ├── final_videos/          # Completed videos ready for posting
│   ├── trends.db              # Trending topics database
│   ├── metrics.db             # Performance metrics
│   └── image_library.db       # Image tracking and metadata
├── docs/                      # Documentation
│   ├── CHARACTER_PROFILES.md  # Character descriptions and voice settings
│   ├── WORKFLOW.md            # Pipeline workflow details
│   ├── INSTAGRAM_SETUP.md     # Instagram API configuration
│   ├── TROUBLESHOOTING.md     # Common issues and solutions
│   └── comfyui_setup.md       # ComfyUI installation guide
├── models/                    # AI models (not in version control)
│   ├── dreambooth_butcher/    # Custom trained Butcher model
│   ├── wan2.2/                # Wan 2.2 video generation models
│   ├── sadtalker/             # SadTalker animation checkpoints
│   └── wav2lip/               # Wav2Lip animation checkpoints
├── scripts/                   # Core pipeline scripts
│   ├── fetch_trends.py        # Fetch trending topics
│   ├── generate_script.py     # Generate dialogue scripts
│   ├── generate_audio.py      # Synthesize character voices
│   ├── select_images.py       # Select/generate character images
│   ├── assemble_video.py      # Compile final video
│   ├── animate.py             # Add lip-sync animation
│   ├── post_instagram.py      # Post to Instagram
│   ├── train_dreambooth.py    # Train custom character models
│   └── utils/                 # Utility modules
├── tests/                     # Unit and integration tests
├── training/                  # Training data and configs
│   ├── butcher/               # Butcher training images (23 images)
│   └── config/                # Training configurations
├── comfyui/                   # ComfyUI workflows and integration
├── n8n/                       # n8n automation workflows
├── Makefile                   # Common commands (30+ shortcuts)
├── Dockerfile                 # Container deployment
├── docker-compose.yml         # Multi-service orchestration
├── PROJECT_STATUS.md          # Detailed project status
├── TODO.md                    # Prioritized task list
├── QUICKSTART.md              # 15-minute setup guide
└── README.md                  # This file
```

## Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 15 minutes
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Detailed project status and component completion
- **[TODO.md](TODO.md)** - Prioritized task list and roadmap
- **[docs/CHARACTER_PROFILES.md](docs/CHARACTER_PROFILES.md)** - Character descriptions and ElevenLabs voice settings
- **[docs/WORKFLOW.md](docs/WORKFLOW.md)** - Detailed pipeline workflow
- **[docs/INSTAGRAM_SETUP.md](docs/INSTAGRAM_SETUP.md)** - Instagram API configuration guide
- **[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## Common Commands

The `Makefile` provides 30+ convenient shortcuts:

```bash
# Setup and validation
make setup                # Initial setup (install + init databases)
make check-env            # Validate environment configuration

# Pipeline operations
make run-pipeline         # Run complete video generation pipeline
make fetch-trends         # Fetch latest trending topics
make clean                # Clean temporary files

# Testing
make test                 # Run all tests
make test-comfyui         # Test ComfyUI integration
make test-instagram       # Test Instagram posting (dry-run)

# Training
make train-butcher        # Train Butcher DreamBooth model

# Maintenance
make backup-db            # Backup all databases
make stats                # Show project statistics

# Help
make help                 # Show all available commands
```

## Advanced Features

### Custom Character Training

Train a custom DreamBooth model for consistent character generation:

```bash
# 1. Add training images (15-25 high-quality photos)
cp /path/to/images/*.jpg training/butcher/images/

# 2. Train model (~15-20 minutes on RTX 4070)
python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

# The trained model will be saved to models/dreambooth_butcher/
```

**Training Tips:**
- Uses Stable Diffusion 2.1 as base model
- Requires HuggingFace token (HF_TOKEN in .env)
- GPU with 12GB+ VRAM recommended
- Training generates 200 class images for prior preservation
- Model trained for 800 steps (configurable in butcher_config.yaml)

See [docs/IMAGE_GENERATION.md](docs/IMAGE_GENERATION.md) for details.

### Animation Pipeline

Add lip-sync animation to your videos:

```bash
# Download animation models first
# See docs/TROUBLESHOOTING.md for download links

# Animate a video
python scripts/animate.py \
  --timeline data/audio/episode_*/timeline.json \
  --images data/image_selections.json \
  --method wav2lip
```

### Instagram Automation

Automatically post videos to Instagram:

```bash
# 1. Configure Instagram API (see docs/INSTAGRAM_SETUP.md)

# 2. Test posting (dry-run)
python scripts/post_instagram.py data/final_videos/episode_*.mp4 --dry-run

# 3. Post for real
python scripts/post_instagram.py data/final_videos/episode_*.mp4
```

### Workflow Automation

Automate the complete pipeline with n8n:

```bash
# Install n8n
npm install -g n8n

# Import workflows
n8n import:workflow --input=n8n/workflows/main_pipeline.json

# Set up daily automation
# (See n8n documentation for scheduling)
```

## Performance Optimizations

The project includes several performance optimizations:

- **Parallel audio generation** - 3x faster with ThreadPoolExecutor
- **Batch database queries** - 10x faster with SQL VALUES clause
- **ComfyUI image caching** - 150x faster with SHA-256 hash-based caching
- **Model preloading** - 800x faster with singleton pattern
- **Checkpoint/resume** - Resume long-running operations from last checkpoint

## Project Status

**Current Status:** 🟡 Core Complete, Testing Phase

- ✅ Core pipeline implemented and tested
- ✅ Security and validation complete
- ✅ Performance optimizations implemented
- ✅ Comprehensive documentation
- 🟡 DreamBooth training ready (needs base model download)
- 🟡 Animation pipeline implemented (needs model downloads)
- 🟡 Instagram posting ready (needs API configuration)
- 🔴 End-to-end testing in progress

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed component status and [TODO.md](TODO.md) for prioritized tasks.

## Troubleshooting

### Common Issues

**"No module named 'elevenlabs'"**
```bash
pip install elevenlabs
```

**"CUDA out of memory"**
- Reduce batch size: `ANIMATION_BATCH_SIZE=4` in `.env`
- Use lower quality: `ANIMATION_QUALITY=medium` in `.env`
- Close other GPU applications

**"CUDA not available" or "torch.cuda.is_available() returns False"**
```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is working
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

**"401 Unauthorized" when downloading models from HuggingFace**
- Create a token at https://huggingface.co/settings/tokens
- Add to `config/.env`: `HF_TOKEN=hf_your_token_here`
- For some models, you may need to accept the license on the model's HuggingFace page

**"API key not found"**
```bash
# Verify .env file exists and has correct format
cat config/.env | grep API_KEY
```

**"Database is locked"**
```bash
# Close any other scripts accessing the database
# Or backup and reinitialize:
make backup-db
python scripts/init_databases.py --reset
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for more solutions.

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Acknowledgments

- **Anthropic Claude** - For script generation and creative assistance
- **ElevenLabs** - For high-quality voice synthesis
- **Stability AI** - For Stable Diffusion base models
- **ComfyUI** - For flexible image generation workflows
- **SadTalker & Wav2Lip** - For lip-sync animation

## Support

- 📖 **Documentation:** See `docs/` directory
- 💬 **Issues:** https://github.com/stewartburton/ExcuseMyFrench/issues
- 📧 **Discussions:** https://github.com/stewartburton/ExcuseMyFrench/discussions

---

Made with ❤️ and 🤖 | Happy creating! 🎬🐕🐿️
