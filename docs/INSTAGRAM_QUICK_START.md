# Instagram Posting - Quick Start Guide

Get up and running with Instagram posting in 5 minutes!

## Quick Setup

1. **Install dependencies:**
```bash
pip install requests
```

2. **Configure credentials in `config/.env`:**
```env
META_ACCESS_TOKEN=your-access-token-here
INSTAGRAM_USER_ID=your-user-id-here
INSTAGRAM_POST_ENABLED=true
```

3. **Test the setup:**
```bash
python scripts/test_instagram_posting.py
```

## Basic Usage

### Post a video
```bash
python scripts/post_instagram.py data/final_videos/video.mp4
```

### Post with custom caption
```bash
python scripts/post_instagram.py video.mp4 --caption "Check this out! #funny"
```

### Preview without posting (dry run)
```bash
python scripts/post_instagram.py video.mp4 --dry-run
```

### Check posting status
```bash
python scripts/post_instagram.py --status
```

## Common Commands

### Caption Generation

Generate a caption from a video:
```bash
python scripts/generate_caption.py video.mp4
```

Generate from script file:
```bash
python scripts/generate_caption.py --script data/scripts/episode_123.json
```

### Queue Management

Add video to queue:
```bash
python scripts/post_instagram.py video.mp4 --queue
```

Process queue (post 1 video):
```bash
python scripts/post_instagram.py --process-queue
```

Process multiple from queue:
```bash
python scripts/post_instagram.py --process-queue --max-posts 3
```

## Complete Workflow

End-to-end video creation and posting:

```bash
# 1. Generate script
python scripts/generate_script.py

# 2. Generate audio
python scripts/generate_audio.py --script data/scripts/episode_123.json

# 3. Select images
python scripts/select_images.py --timeline data/audio/episode_123/timeline.json

# 4. Assemble video
python scripts/assemble_video.py \
  --timeline data/audio/episode_123/timeline.json \
  --images data/image_selections.json

# 5. Generate caption
python scripts/generate_caption.py data/final_videos/episode_123.mp4

# 6. Post to Instagram
python scripts/post_instagram.py data/final_videos/episode_123.mp4
```

## Troubleshooting

### "Access token is invalid"
- Generate a new token at https://developers.facebook.com/tools/explorer/
- Make sure you have these permissions:
  - instagram_basic
  - instagram_content_publish
  - pages_read_engagement

### "Rate limit exceeded"
- Wait 1 hour before posting again
- Use `--queue` to schedule posts
- The script automatically retries with backoff

### "File not found"
- Check the video path is correct
- Make sure the video exists in `data/final_videos/`
- Use absolute or relative paths

### Need Help?

1. Run tests: `python scripts/test_instagram_posting.py`
2. Check logs: Look in console output for errors
3. Review setup: See `docs/INSTAGRAM_SETUP.md` for detailed instructions
4. Verify status: `python scripts/post_instagram.py --status`

## Configuration Options

In `config/.env`:

```env
# Required
META_ACCESS_TOKEN=your-token
INSTAGRAM_USER_ID=your-id

# Optional
INSTAGRAM_POST_ENABLED=true
INSTAGRAM_HASHTAGS=#custom #hashtags #here
INSTAGRAM_CAPTION_TEMPLATE={hook}\n\n{content}\n\n{hashtags}

# API Settings
API_MAX_RETRIES=3
API_RETRY_DELAY=2
```

## Tips

- **Always test first**: Use `--dry-run` before posting
- **Monitor status**: Check `--status` regularly
- **Use queue**: Avoid rate limits by queuing posts
- **Custom captions**: Use `--caption` for special posts
- **Track history**: Check `data/instagram/posted_history.json`

## Next Steps

Once posting works:

1. Set up automated posting (cron job or Task Scheduler)
2. Monitor analytics in `data/instagram/analytics.json`
3. Adjust hashtags based on performance
4. Schedule regular posts via the queue system

Happy posting! 🎬
