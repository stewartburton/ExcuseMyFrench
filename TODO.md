# TODO - Excuse My French

**Last Updated:** October 7, 2025

This document tracks all remaining tasks to complete the ExcuseMyFrench project. Tasks are organized by priority and category.

---

## 🔴 Critical (Blocking Progress)

### Model Setup

- [ ] **Download Stable Diffusion v1.5 Base Model**
  - **Priority:** HIGHEST - Blocks DreamBooth training
  - **Estimated Time:** 30 minutes
  - **Size:** ~4GB
  - **Action:**
    ```bash
    # Option 1: Using HuggingFace CLI
    huggingface-cli login
    huggingface-cli download runwayml/stable-diffusion-v1-5

    # Option 2: Will auto-download on first run (but currently failing)
    python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
    ```
  - **Details:** The model will be downloaded to HuggingFace cache directory
  - **Blocker:** DreamBooth training cannot start without this

- [ ] **Train Butcher DreamBooth Model**
  - **Priority:** HIGH - Required for consistent character generation
  - **Estimated Time:** 2-3 hours (RTX 4070)
  - **Dependencies:** Stable Diffusion base model downloaded
  - **Action:**
    ```bash
    python scripts/train_dreambooth.py --config training/config/butcher_config.yaml
    ```
  - **Status:** Training data ready (23 validated images)
  - **Output:** Custom model at `models/dreambooth_butcher/`

### API Configuration

- [ ] **Set Up ElevenLabs API Keys**
  - **Priority:** HIGH - Required for audio generation
  - **Estimated Time:** 15 minutes
  - **Action:**
    1. Sign up at https://elevenlabs.io/
    2. Get API key from dashboard
    3. Add to `config/.env`: `ELEVENLABS_API_KEY=your_key`
  - **Note:** Free tier includes 10,000 characters/month

- [ ] **Create & Configure Character Voices**
  - **Priority:** HIGH - Required for audio generation
  - **Estimated Time:** 30 minutes
  - **Action:**
    1. Go to ElevenLabs voice library
    2. Create or clone voices for Butcher and Nutsy
    3. Use descriptions from `docs/CHARACTER_PROFILES.md`
    4. Get voice IDs and add to `.env`:
       ```
       ELEVENLABS_VOICE_BUTCHER=your_butcher_voice_id
       ELEVENLABS_VOICE_NUTSY=your_nutsy_voice_id
       ```
  - **Testing:** Generate sample audio to verify voice quality

- [ ] **Configure LLM API Key**
  - **Priority:** HIGH - Required for script generation
  - **Estimated Time:** 5 minutes
  - **Action:** Add one of the following to `config/.env`:
    ```bash
    OPENAI_API_KEY=sk-...
    # OR
    ANTHROPIC_API_KEY=sk-ant-...
    ```
  - **Note:** OpenAI GPT-4 recommended for best script quality

---

## 🟡 High Priority (Core Functionality)

### Testing & Validation

- [ ] **Run Environment Validation**
  - **Priority:** HIGH
  - **Estimated Time:** 5 minutes
  - **Action:**
    ```bash
    python scripts/validate_env.py --verbose
    # OR
    make check-env
    ```
  - **Expected:** All checks should pass before pipeline testing

- [ ] **Test Trending Topics Fetch**
  - **Priority:** HIGH
  - **Estimated Time:** 5 minutes
  - **Action:**
    ```bash
    python scripts/fetch_trends.py --days 7 --limit 10
    make fetch-trends
    ```
  - **Verify:** Check `data/trends.db` for entries

- [ ] **Test Script Generation**
  - **Priority:** HIGH
  - **Estimated Time:** 2-3 minutes
  - **Dependencies:** LLM API key configured
  - **Action:**
    ```bash
    python scripts/generate_script.py --from-trends --days 3
    ```
  - **Verify:** Check `data/scripts/` for generated episode JSON

- [ ] **Test Audio Generation**
  - **Priority:** HIGH
  - **Estimated Time:** 3-5 minutes per episode
  - **Dependencies:** ElevenLabs API + voice IDs configured
  - **Action:**
    ```bash
    python scripts/generate_audio.py data/scripts/episode_*.json
    ```
  - **Verify:** Check `data/audio/` for MP3 files and timeline.json

- [ ] **Test Image Selection**
  - **Priority:** HIGH
  - **Estimated Time:** 1-2 minutes
  - **Action:**
    ```bash
    python scripts/select_images.py data/scripts/episode_*.json
    ```
  - **Verify:** Check `data/image_selections.json` generated
  - **Note:** Will use placeholder images if no custom images available

- [ ] **Test Video Assembly**
  - **Priority:** HIGH
  - **Estimated Time:** 1-2 minutes per video
  - **Action:**
    ```bash
    python scripts/assemble_video.py \
      --timeline data/audio/episode_*/timeline.json \
      --images data/image_selections.json
    ```
  - **Verify:** Check `data/final_videos/` for output MP4

- [ ] **Run Full End-to-End Pipeline**
  - **Priority:** HIGH - Validates entire workflow
  - **Estimated Time:** 10-15 minutes
  - **Dependencies:** All API keys configured
  - **Action:**
    ```bash
    make run-pipeline
    # OR
    python scripts/run_pipeline.py
    ```
  - **Success Criteria:**
    - No errors during execution
    - Final video generated in `data/final_videos/`
    - Video plays correctly
    - Audio and images synced properly

### Character Image Generation

- [ ] **Test DreamBooth Model (After Training)**
  - **Priority:** HIGH
  - **Estimated Time:** 2-3 minutes per image
  - **Dependencies:** DreamBooth training completed
  - **Action:**
    ```bash
    python scripts/generate_character_image.py \
      --character butcher \
      --emotion happy \
      --prompt "a photo of sks dog, happy expression, professional photography"
    ```
  - **Verify:** Generated image looks like Butcher
  - **Note:** May need to adjust prompt or retrain if quality is poor

---

## 🟢 Medium Priority (Enhanced Features)

### Animation Pipeline

- [X] **Download Wav2Lip Checkpoints** ✅
  - **Priority:** MEDIUM
  - **Estimated Time:** 10 minutes
  - **Size:** ~366MB
  - **Status:** COMPLETED
  - **Files Downloaded:**
    - ✅ s3fd.pth (86MB) - Face detection
    - ✅ Wav2Lip-SD-GAN.pt (140MB) - High quality lip-sync
    - ✅ Wav2Lip-SD-NOGAN.pt (140MB) - Faster lip-sync
  - **Location:** `models/wav2lip/`
  - **Note:** Downloaded SD (Stable Diffusion enhanced) versions, not older `wav2lip*.pth` versions
  - **Full Instructions:** `docs/MODEL_DOWNLOADS.md`

- [X] **Download SadTalker Checkpoints** ✅
  - **Priority:** MEDIUM
  - **Estimated Time:** 15 minutes
  - **Size:** ~1.4GB
  - **Status:** COMPLETED
  - **Files Downloaded:**
    - ✅ SadTalker_V0.0.2_256.safetensors (692MB)
    - ✅ SadTalker_V0.0.2_512.safetensors (692MB)
    - ✅ alignment_WFLW_4HG.pth (185MB)
    - ✅ GFPGANv1.4.pth (333MB)
    - ✅ headpose_hopenet.pth (92MB)
  - **Location:** `models/sadtalker/checkpoints/` and `models/sadtalker/gfpgan/weights/`
  - **Note:** Using V0.0.2 models (newer, recommended version)
  - **Full Instructions:** `docs/MODEL_DOWNLOADS.md`

- [ ] **Test Animation Pipeline**
  - **Priority:** MEDIUM
  - **Estimated Time:** 5-10 minutes per video
  - **Dependencies:** Animation models downloaded
  - **Action:**
    ```bash
    python scripts/animate.py \
      --timeline data/audio/episode_*/timeline.json \
      --images data/image_selections.json \
      --method wav2lip
    ```
  - **Verify:** Animated video with lip-sync in `data/animated/`

- [ ] **Test Checkpoint/Resume Functionality**
  - **Priority:** MEDIUM
  - **Action:**
    1. Start animation batch
    2. Cancel mid-process (Ctrl+C)
    3. Restart with same command
    4. Verify it resumes from checkpoint
  - **Expected:** Skips already-processed frames

### Instagram Integration

- [ ] **Set Up Instagram API Access**
  - **Priority:** MEDIUM
  - **Estimated Time:** 1-2 hours
  - **Action:** Follow guide in `docs/INSTAGRAM_SETUP.md`
  - **Steps:**
    1. Create Meta Developer account
    2. Create Facebook App
    3. Configure Instagram Basic Display
    4. Get long-lived access token
    5. Add to `.env`:
       ```
       META_ACCESS_TOKEN=your_token
       INSTAGRAM_USER_ID=your_user_id
       ```

- [ ] **Test Instagram Posting (Dry Run)**
  - **Priority:** MEDIUM
  - **Estimated Time:** 2 minutes
  - **Dependencies:** Instagram API configured
  - **Action:**
    ```bash
    python scripts/post_instagram.py \
      data/final_videos/episode_*.mp4 \
      --dry-run
    ```
  - **Verify:** No errors, caption generated correctly

- [ ] **Test Instagram Posting (Real)**
  - **Priority:** MEDIUM
  - **Estimated Time:** 3-5 minutes
  - **Dependencies:** Dry run successful
  - **Action:**
    ```bash
    python scripts/post_instagram.py data/final_videos/episode_*.mp4
    ```
  - **Verify:** Video posted to Instagram, check post URL

- [ ] **Test Token Refresh Mechanism**
  - **Priority:** MEDIUM
  - **Action:**
    1. Set token expiry to near-future date in `.env`
    2. Run posting script
    3. Verify automatic token refresh occurs
  - **Expected:** Warning logged, token refreshed automatically

### ComfyUI Integration (Optional)

- [ ] **Install ComfyUI**
  - **Priority:** LOW (optional for v1.0)
  - **Estimated Time:** 30 minutes
  - **Action:**
    ```bash
    make setup-comfyui
    # OR manually:
    cd comfyui
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd ComfyUI && pip install -r requirements.txt
    ```
  - **Note:** Only needed if generating images with Stable Diffusion workflows

- [ ] **Test ComfyUI Integration**
  - **Priority:** LOW
  - **Estimated Time:** 5 minutes
  - **Dependencies:** ComfyUI installed, DreamBooth model trained
  - **Action:**
    ```bash
    python scripts/test_comfyui.py
    ```
  - **Verify:** Sample image generated successfully

- [ ] **Set Up ComfyUI Workflows**
  - **Priority:** LOW
  - **Action:**
    1. Copy workflows from `comfyui/workflows/` to ComfyUI directory
    2. Test each workflow manually in ComfyUI web UI
    3. Verify parameterization works with `scripts/utils/workflow_params.py`

### Video Generation (Optional)

- [X] **Download Wan 2.2 Models** ✅
  - **Priority:** LOW (optional for v1.0)
  - **Estimated Time:** 1-3 hours
  - **Size:** ~32GB (TI2V-5B model)
  - **Status:** COMPLETED
  - **Files Downloaded:**
    - ✅ Wan2.2_VAE.pth (2.7GB)
    - ✅ diffusion_pytorch_model-00001-of-00003.safetensors (9.2GB)
    - ✅ diffusion_pytorch_model-00002-of-00003.safetensors (9.4GB)
    - ✅ diffusion_pytorch_model-00003-of-00003.safetensors (171MB)
    - ✅ models_t5_umt5-xxl-enc-bf16.pth (11GB) - T5 text encoder
    - ✅ Config files, index files, examples
  - **Location:** `models/wan2.2/TI2V-5B/`
  - **Capabilities:** 720P@24fps text-to-video and image-to-video generation
  - **Full Instructions:** `docs/MODEL_DOWNLOADS.md`

---

## 🔵 Low Priority (Nice to Have)

### Advanced Testing

- [ ] **Write Integration Tests**
  - **Priority:** LOW
  - **Estimated Time:** 4-6 hours
  - **Files to Create:**
    - `tests/test_integration.py` - End-to-end tests
    - `tests/test_rate_limiting.py` - Rate limit tests
    - `tests/test_checkpoint.py` - Checkpoint/resume tests
  - **Action:**
    ```bash
    pytest tests/ -v --cov=scripts
    ```

- [ ] **Write Performance Benchmarks**
  - **Priority:** LOW
  - **Estimated Time:** 2-3 hours
  - **Action:**
    1. Create `scripts/benchmark.py`
    2. Measure each pipeline stage
    3. Compare with/without optimizations
    4. Document results in `docs/PERFORMANCE.md`

- [ ] **Run Load Testing**
  - **Priority:** LOW
  - **Action:**
    1. Generate 100 videos in batch
    2. Monitor memory usage
    3. Check for memory leaks
    4. Verify checkpoint/resume under load

### Documentation

- [ ] **Create TESTING.md Documentation**
  - **Priority:** LOW
  - **Content:**
    - Unit testing guidelines
    - Integration testing procedures
    - Performance testing methodology
    - CI/CD integration (if applicable)

- [ ] **Create DEPLOYMENT.md Documentation**
  - **Priority:** LOW
  - **Content:**
    - Production deployment guide
    - Server requirements
    - Security hardening
    - Monitoring and logging
    - Backup procedures

- [ ] **Create API.md Documentation**
  - **Priority:** LOW
  - **Content:**
    - Internal script APIs
    - Database schema reference
    - Workflow JSON format
    - Extension points for custom features

- [ ] **Update README with Badges**
  - **Priority:** LOW
  - **Action:** Add badges for:
    - Build status (if CI/CD set up)
    - Test coverage
    - License
    - Python version
    - Contributors

### Automation

- [ ] **Set Up n8n Orchestration**
  - **Priority:** LOW
  - **Estimated Time:** 3-4 hours
  - **Action:**
    ```bash
    npm install -g n8n
    n8n import:workflow --input=n8n/workflows/main_pipeline.json
    ```
  - **Configuration:**
    1. Set up schedule (e.g., daily at 9 AM)
    2. Configure error notifications
    3. Test workflow execution

- [ ] **Create Cron Job for Daily Posting**
  - **Priority:** LOW
  - **Dependencies:** Full pipeline tested and working
  - **Action:**
    ```bash
    crontab -e
    # Add: 0 9 * * * cd /path/to/ExcuseMyFrench && make run-pipeline
    ```

- [ ] **Set Up Monitoring & Alerts**
  - **Priority:** LOW
  - **Tools to Consider:**
    - Prometheus + Grafana for metrics
    - Sentry for error tracking
    - Custom webhook for notifications
  - **Metrics to Track:**
    - Pipeline success/failure rate
    - API costs
    - Video generation time
    - Instagram engagement

### Advanced Features

- [ ] **Train Nutsy Character Model**
  - **Priority:** LOW
  - **Estimated Time:** 2-3 hours training
  - **Action:**
    1. Collect 15-20 squirrel images
    2. Create `training/nutsy/images/` directory
    3. Copy `training/config/butcher_config.yaml` to `nutsy_config.yaml`
    4. Update config for Nutsy
    5. Run training
  - **Note:** Currently using generic squirrel images, custom model would improve consistency

- [ ] **Add Music Generation**
  - **Priority:** LOW
  - **Options:**
    - Mubert API integration
    - Soundverse API integration
    - Local music library
  - **Action:**
    1. Choose service and get API key
    2. Update `scripts/assemble_video.py` to add background music
    3. Implement audio mixing with FFmpeg

- [ ] **Implement A/B Testing**
  - **Priority:** LOW
  - **Features to Test:**
    - Script styles (sarcastic vs playful)
    - Caption formats
    - Hashtag strategies
    - Posting times
  - **Action:**
    1. Create variants of content
    2. Track performance metrics
    3. Analyze results
    4. Optimize based on data

- [ ] **Add Multi-Platform Support**
  - **Priority:** LOW
  - **Platforms:**
    - TikTok API
    - YouTube Shorts API
    - Twitter/X video API
  - **Action:**
    1. Research APIs for each platform
    2. Implement posting scripts
    3. Adapt video formats (aspect ratios, durations)

- [ ] **Create Analytics Dashboard**
  - **Priority:** LOW
  - **Metrics to Display:**
    - Video performance (views, likes, comments)
    - API costs
    - Generation times
    - Trending topic effectiveness
  - **Tools:** Streamlit or Dash for Python dashboard

---

## ⚙️ Maintenance & Operations

### Regular Tasks

- [ ] **Weekly: Review API Costs**
  - Check OpenAI/Anthropic usage
  - Check ElevenLabs character usage
  - Adjust limits if needed

- [ ] **Weekly: Backup Databases**
  - **Action:**
    ```bash
    make backup-db
    ```
  - Store backups offsite

- [ ] **Monthly: Update Dependencies**
  - **Action:**
    ```bash
    pip list --outdated
    pip install --upgrade [package]
    ```
  - Test after updates

- [ ] **Monthly: Review and Clean Generated Content**
  - Archive old videos
  - Clean temporary files
  - Free up disk space

### Security

- [ ] **Rotate API Keys**
  - Schedule: Every 90 days
  - Update in `.env` file
  - Update in any CI/CD secrets

- [ ] **Review Access Logs**
  - Check for suspicious API usage
  - Monitor Instagram API rate limits
  - Review error logs

- [ ] **Update Security Dependencies**
  - Run `pip audit` or similar
  - Fix any vulnerabilities
  - Update Dockerfile if needed

---

## 📊 Progress Tracking

### Overall Project Status

- **Core Functionality:** 85% complete
- **Testing:** 20% complete
- **Documentation:** 90% complete
- **Automation:** 30% complete
- **Advanced Features:** 10% complete

### Current Sprint (Next 7 Days)

**Focus:** Get basic pipeline working end-to-end

1. Download Stable Diffusion base model ⬜
2. Train Butcher DreamBooth model ⬜
3. Configure all required API keys ⬜
4. Run full end-to-end pipeline test ⬜
5. Generate first production-ready video ⬜

### Next Sprint (Days 8-14)

**Focus:** Advanced features and automation

1. Set up animation pipeline ⬜
2. Configure Instagram posting ⬜
3. Test batch video generation ⬜
4. Set up automated scheduling ⬜

---

## 🎯 Success Criteria

### Minimum Viable Product (MVP)

- [X] All core scripts implemented
- [ ] API keys configured
- [ ] DreamBooth model trained
- [ ] Full pipeline tested successfully
- [ ] At least 3 videos generated and reviewed
- [ ] Documentation complete

### Version 1.0 Release

- [ ] MVP complete
- [ ] Animation pipeline working
- [ ] Instagram auto-posting configured
- [ ] Performance optimizations verified
- [ ] Comprehensive testing completed
- [ ] Production deployment guide written

### Version 2.0 Vision

- [ ] Multi-platform posting (TikTok, YouTube Shorts)
- [ ] Advanced animation (SadTalker + emotion control)
- [ ] Music generation integration
- [ ] Analytics dashboard
- [ ] A/B testing framework
- [ ] Community features (user-submitted topics)

---

## 📝 Notes

### Decisions Made

- Using ElevenLabs for voice (better quality than alternatives)
- Using DreamBooth for character consistency (vs generic Stable Diffusion)
- SQLite for databases (vs PostgreSQL - sufficient for current scale)
- FFmpeg for video assembly (vs MoviePy - faster and more reliable)
- n8n for orchestration (vs Airflow - simpler for this use case)

### Open Questions

- [ ] Which Wan 2.2 model size? (5B vs 14B - depends on VRAM)
- [ ] Animation method? (SadTalker vs Wav2Lip - need to test both)
- [ ] Hosting strategy? (Self-hosted vs cloud - depends on budget)
- [ ] Posting frequency? (Daily vs weekly - depends on content quality)

### Lessons Learned

- Parallel processing critical for audio generation (3x speedup)
- Batch database queries essential for performance (10x speedup)
- Checkpoint/resume needed for long-running operations
- Path validation prevents security issues
- Rate limiting must be proactive, not reactive

---

## 🤝 Getting Help

If stuck on any task:

1. Check `docs/TROUBLESHOOTING.md`
2. Review relevant documentation in `docs/`
3. Check GitHub issues: https://github.com/stewartburton/ExcuseMyFrench/issues
4. Ask for help in project discussions

---

**Last Updated:** October 7, 2025
**Next Review:** After completing current sprint tasks
