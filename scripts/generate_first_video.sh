#!/bin/bash
# Generate First Video - Complete End-to-End Workflow
#
# This script walks you through generating your first complete video
# with Butcher and Nutsy, from script generation to final video assembly.
#
# Prerequisites:
# - DreamBooth training completed
# - API keys configured in config/.env
# - FFmpeg installed
#
# Usage:
#   bash scripts/generate_first_video.sh
#   bash scripts/generate_first_video.sh --skip-images  # Use existing images
#   bash scripts/generate_first_video.sh --topic "AI"   # Custom topic

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SKIP_IMAGES=false
CUSTOM_TOPIC=""
EPISODE_ID="first_video_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-images)
      SKIP_IMAGES=true
      shift
      ;;
    --topic)
      CUSTOM_TOPIC="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --skip-images    Skip image generation, use existing images"
      echo "  --topic TOPIC    Use custom topic instead of trending topics"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Banner
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}                   GENERATE YOUR FIRST VIDEO!${NC}"
echo -e "${BLUE}                   Butcher & Nutsy in Action${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python installed"

# Check FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg not found${NC}"
    echo "Install with: choco install ffmpeg -y (Windows admin PowerShell)"
    exit 1
fi
echo -e "${GREEN}✓${NC} FFmpeg installed"

# Check .env file
if [ ! -f "config/.env" ]; then
    echo -e "${RED}Error: config/.env not found${NC}"
    echo "Create it from: config/.env.example"
    exit 1
fi
echo -e "${GREEN}✓${NC} Environment config found"

# Check DreamBooth model
if [ ! -d "models/dreambooth_butcher" ]; then
    echo -e "${RED}Error: DreamBooth model not found at models/dreambooth_butcher/${NC}"
    echo "Please complete training first: python scripts/train_dreambooth.py --config training/config/butcher_config.yaml"
    exit 1
fi
echo -e "${GREEN}✓${NC} DreamBooth model found"

echo ""
echo -e "${GREEN}All prerequisites met!${NC}"
echo ""

# Step 1: Generate Script
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}STEP 1: Generate Script${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

if [ -n "$CUSTOM_TOPIC" ]; then
    echo -e "Generating script about: ${YELLOW}$CUSTOM_TOPIC${NC}"
    python scripts/generate_script.py \
        --topic "$CUSTOM_TOPIC" \
        --format short \
        --output "data/scripts/$EPISODE_ID.json"
else
    echo "Fetching trending topics..."
    python scripts/fetch_trends.py --days 7 --limit 10 || true

    echo -e "Generating script from trending topics..."
    python scripts/generate_script.py \
        --from-trends \
        --days 3 \
        --output "data/scripts/$EPISODE_ID.json"
fi

if [ ! -f "data/scripts/$EPISODE_ID.json" ]; then
    echo -e "${RED}Error: Script generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Script generated: data/scripts/$EPISODE_ID.json${NC}"
echo ""

# Show script preview
echo -e "${YELLOW}Script Preview:${NC}"
python -c "
import json
with open('data/scripts/$EPISODE_ID.json') as f:
    script = json.load(f)
    print(f\"Title: {script.get('title', 'Untitled')}\" )
    print(f\"Lines: {len(script.get('script', []))}\" )
    print(f\"Topic: {script.get('metadata', {}).get('topic', 'N/A')}\" )
"
echo ""

# Step 2: Generate Audio
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}STEP 2: Generate Character Voices${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

echo "Generating audio with ElevenLabs..."
python scripts/generate_audio.py "data/scripts/$EPISODE_ID.json"

if [ ! -d "data/audio/$EPISODE_ID" ]; then
    echo -e "${RED}Error: Audio generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Audio generated: data/audio/$EPISODE_ID/${NC}"
echo ""

# Show audio info
echo -e "${YELLOW}Audio Files:${NC}"
ls -lh "data/audio/$EPISODE_ID/"*.mp3 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Step 3: Generate Images
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}STEP 3: Generate Character Images${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

if [ "$SKIP_IMAGES" = true ]; then
    echo -e "${YELLOW}Skipping image generation (using existing images)${NC}"
else
    echo "Generating character images with DreamBooth model..."
    echo ""

    # Generate images for common emotions
    for emotion in happy sarcastic grumpy excited; do
        echo -e "${YELLOW}Generating ${emotion} image...${NC}"
        python scripts/generate_character_image.py \
            --character butcher \
            --emotion "$emotion" \
            --prompt "a photo of sks dog, $emotion expression, professional photography" \
            --output "data/images/butcher_${emotion}_001.png" || true
    done

    # Generate Nutsy images
    for emotion in excited curious; do
        echo -e "${YELLOW}Generating Nutsy ${emotion} image...${NC}"
        python scripts/generate_character_image.py \
            --character nutsy \
            --emotion "$emotion" \
            --prompt "a photo of a hyperactive squirrel, $emotion expression, bright eyes" \
            --output "data/images/nutsy_${emotion}_001.png" || true
    done
fi

echo -e "${GREEN}✓ Character images ready${NC}"
echo ""

# Step 4: Select Images for Timeline
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}STEP 4: Select Images for Timeline${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

echo "Matching images to dialogue..."
python scripts/select_images.py "data/scripts/$EPISODE_ID.json"

if [ ! -f "data/image_selections.json" ]; then
    echo -e "${RED}Error: Image selection failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Image selections created: data/image_selections.json${NC}"
echo ""

# Step 5: Assemble Video
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}STEP 5: Assemble Final Video${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

echo "Assembling video with FFmpeg..."
python scripts/assemble_video.py \
    --timeline "data/audio/$EPISODE_ID/timeline.json" \
    --images data/image_selections.json \
    --output "data/final_videos/$EPISODE_ID.mp4"

if [ ! -f "data/final_videos/$EPISODE_ID.mp4" ]; then
    echo -e "${RED}Error: Video assembly failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Video assembled: data/final_videos/$EPISODE_ID.mp4${NC}"
echo ""

# Show video info
echo -e "${YELLOW}Video Information:${NC}"
ffprobe -v error -show_format -show_streams "data/final_videos/$EPISODE_ID.mp4" 2>&1 | grep -E "duration|width|height|codec_name" | head -5
echo ""

# Success Summary
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}                    🎉 SUCCESS! 🎉${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""
echo -e "${GREEN}Your first video has been generated!${NC}"
echo ""
echo -e "${YELLOW}Generated Files:${NC}"
echo "  📝 Script:     data/scripts/$EPISODE_ID.json"
echo "  🔊 Audio:      data/audio/$EPISODE_ID/"
echo "  🖼️  Images:     data/images/"
echo "  🎬 Video:      data/final_videos/$EPISODE_ID.mp4"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Open and watch: data/final_videos/$EPISODE_ID.mp4"
echo "  2. Review quality (audio, images, sync)"
echo "  3. If quality is good:"
echo "     - Generate more videos with: make run-pipeline"
echo "     - Set up Instagram posting: docs/INSTAGRAM_SETUP.md"
echo "  4. If quality needs improvement:"
echo "     - Adjust prompts for image generation"
echo "     - Retrain DreamBooth model if character inconsistent"
echo "     - Check API keys and voice settings"
echo ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}Happy creating! 🎬🐕🐿️${NC}"
echo -e "${BLUE}========================================================================${NC}"
