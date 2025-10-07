# Instagram Posting API Reference

Complete reference for the Instagram posting system.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `post_instagram.py` | Main script for posting videos to Instagram |
| `generate_caption.py` | Generate engaging captions from video content |
| `test_instagram_posting.py` | Test suite for Instagram functionality |
| `full_workflow_example.py` | Complete workflow demonstration |

## InstagramPoster Class

### Constructor

```python
InstagramPoster(dry_run: bool = False)
```

Initialize the Instagram poster.

**Parameters:**
- `dry_run` (bool): If True, simulate posting without actually uploading

**Raises:**
- `ValueError`: If required configuration is missing

**Example:**
```python
from post_instagram import InstagramPoster

# Normal mode
poster = InstagramPoster()

# Dry run mode (testing)
poster = InstagramPoster(dry_run=True)
```

### Methods

#### upload_video()

```python
upload_video(
    video_path: str,
    caption: Optional[str] = None,
    cover_url: Optional[str] = None,
    share_to_feed: bool = True
) -> Optional[str]
```

Upload a video to Instagram as a Reel.

**Parameters:**
- `video_path` (str): Path to video file
- `caption` (str, optional): Video caption. Auto-generated if None
- `cover_url` (str, optional): URL to cover image
- `share_to_feed` (bool): Whether to share to main feed

**Returns:**
- Instagram media ID if successful, None otherwise

**Example:**
```python
media_id = poster.upload_video(
    "data/final_videos/episode_123.mp4",
    caption="Check out this hilarious moment! #funny"
)
```

#### add_to_queue()

```python
add_to_queue(
    video_path: str,
    caption: Optional[str] = None,
    scheduled_time: Optional[str] = None
)
```

Add video to posting queue.

**Parameters:**
- `video_path` (str): Path to video file
- `caption` (str, optional): Video caption
- `scheduled_time` (str, optional): ISO format datetime string

**Example:**
```python
poster.add_to_queue(
    "video.mp4",
    caption="Scheduled post",
    scheduled_time="2024-01-01T18:00:00"
)
```

#### process_queue()

```python
process_queue(max_posts: int = 1)
```

Process videos in the queue.

**Parameters:**
- `max_posts` (int): Maximum number of videos to post

**Example:**
```python
# Post one video from queue
poster.process_queue(max_posts=1)

# Post up to 5 videos
poster.process_queue(max_posts=5)
```

#### get_status()

```python
get_status() -> Dict
```

Get current posting status and statistics.

**Returns:**
- Dictionary with status information

**Example:**
```python
status = poster.get_status()
print(f"Total posts: {status['total_posts']}")
print(f"Successful: {status['successful_posts']}")
```

## CaptionGenerator Class

### Constructor

```python
CaptionGenerator()
```

Initialize the caption generator.

**Example:**
```python
from generate_caption import CaptionGenerator

generator = CaptionGenerator()
```

### Methods

#### generate_from_video()

```python
generate_from_video(video_path: str) -> str
```

Generate caption from video file.

**Parameters:**
- `video_path` (str): Path to video file

**Returns:**
- Generated caption with hashtags

**Example:**
```python
caption = generator.generate_from_video("video.mp4")
```

#### generate_from_script()

```python
generate_from_script(script_lines: List[dict]) -> str
```

Generate caption from script lines.

**Parameters:**
- `script_lines` (list): List of script line dictionaries

**Returns:**
- Generated caption with hashtags

**Example:**
```python
script = [
    {
        "character": "Butcher",
        "line": "Did you hear about AI?",
        "emotion": "sarcastic"
    }
]
caption = generator.generate_from_script(script)
```

#### generate_from_script_file()

```python
generate_from_script_file(script_path: str) -> str
```

Generate caption from script JSON file.

**Parameters:**
- `script_path` (str): Path to script JSON file

**Returns:**
- Generated caption with hashtags

**Example:**
```python
caption = generator.generate_from_script_file(
    "data/scripts/episode_123.json"
)
```

## Command-Line Interface

### post_instagram.py

```bash
python scripts/post_instagram.py [video] [options]
```

**Arguments:**
- `video`: Path to video file (optional, required unless using --status or --process-queue)

**Options:**
- `--caption TEXT`: Custom caption for the video
- `--cover-url URL`: URL to cover image
- `--no-feed`: Don't share to main feed (Reels only)
- `--queue`: Add to queue instead of posting immediately
- `--process-queue`: Process videos in the queue
- `--max-posts N`: Maximum number of videos to post from queue (default: 1)
- `--status`: Show posting status and statistics
- `--dry-run`: Preview actions without posting
- `--verbose`: Enable verbose logging

**Examples:**

Post a video:
```bash
python scripts/post_instagram.py data/final_videos/video.mp4
```

Post with custom caption:
```bash
python scripts/post_instagram.py video.mp4 --caption "Amazing content!"
```

Dry run:
```bash
python scripts/post_instagram.py video.mp4 --dry-run
```

Add to queue:
```bash
python scripts/post_instagram.py video.mp4 --queue
```

Process queue:
```bash
python scripts/post_instagram.py --process-queue --max-posts 3
```

Check status:
```bash
python scripts/post_instagram.py --status
```

### generate_caption.py

```bash
python scripts/generate_caption.py [input] [options]
```

**Arguments:**
- `input`: Path to video file or script JSON

**Options:**
- `--script`: Input is a script JSON file (not a video)
- `--hashtags TEXT`: Custom hashtags to use
- `--output FILE`: Save caption to file
- `--verbose`: Enable verbose logging

**Examples:**

Generate from video:
```bash
python scripts/generate_caption.py video.mp4
```

Generate from script:
```bash
python scripts/generate_caption.py --script data/scripts/episode.json
```

Custom hashtags:
```bash
python scripts/generate_caption.py video.mp4 --hashtags "#custom #tags"
```

Save to file:
```bash
python scripts/generate_caption.py video.mp4 --output caption.txt
```

### test_instagram_posting.py

```bash
python scripts/test_instagram_posting.py
```

Runs a comprehensive test suite of all Instagram functionality.

No arguments needed - just run it to verify your setup.

### full_workflow_example.py

```bash
python scripts/full_workflow_example.py [options]
```

**Options:**
- `--topics TOPIC [TOPIC ...]`: Topics to generate script about
- `--script FILE`: Use existing script file
- `--music FILE`: Path to background music file
- `--dry-run`: Preview posting without uploading
- `--no-post`: Skip Instagram posting step
- `--queue`: Add to posting queue instead of posting immediately
- `--verbose`: Enable verbose logging

**Examples:**

Full workflow (dry run):
```bash
python scripts/full_workflow_example.py --dry-run
```

Custom topics:
```bash
python scripts/full_workflow_example.py --topics "AI" "robots"
```

Use existing script:
```bash
python scripts/full_workflow_example.py --script data/scripts/episode.json
```

Generate video only:
```bash
python scripts/full_workflow_example.py --no-post
```

## Configuration

### Environment Variables

All configuration is in `config/.env`:

```env
# Required
META_ACCESS_TOKEN=your-access-token
INSTAGRAM_USER_ID=your-user-id

# Optional
INSTAGRAM_POST_ENABLED=true
INSTAGRAM_HASHTAGS=#frenchbulldog #comedy #funny
INSTAGRAM_CAPTION_TEMPLATE={hook}\n\n{content}\n\n{hashtags}

# API Settings
API_MAX_RETRIES=3
API_RETRY_DELAY=2
```

### Data Files

Located in `data/instagram/`:

#### posted_history.json

Tracks posted videos to prevent duplicates.

```json
{
  "posted_videos": [
    {
      "video_path": "path/to/video.mp4",
      "file_hash": "sha256_hash",
      "media_id": "instagram_media_id",
      "caption": "video caption...",
      "status": "success",
      "posted_at": "2024-01-01T12:00:00"
    }
  ]
}
```

#### queue.json

Stores videos waiting to be posted.

```json
{
  "queue": [
    {
      "video_path": "path/to/video.mp4",
      "caption": "optional caption",
      "scheduled_time": "2024-01-01T18:00:00",
      "added_at": "2024-01-01T10:00:00",
      "status": "pending"
    }
  ]
}
```

#### analytics.json

Tracks posting statistics.

```json
{
  "total_posts": 10,
  "successful_posts": 9,
  "failed_posts": 1,
  "last_post_time": "2024-01-01T12:00:00",
  "posts": [...]
}
```

## Error Handling

The system includes comprehensive error handling:

### Rate Limiting

- Automatic retry with exponential backoff
- Configurable retry attempts and delays
- Queue system to spread out posts

### Duplicate Prevention

- SHA-256 hashing of video files
- History tracking in `posted_history.json`
- Automatic duplicate detection before posting

### API Errors

All API errors are logged with details:
- 429: Rate limit exceeded (automatic retry)
- 400: Bad request (check video format)
- 401: Invalid token (refresh your token)
- 500: Server error (automatic retry)

## Best Practices

### Testing

1. Always use `--dry-run` first:
```python
poster = InstagramPoster(dry_run=True)
```

2. Run the test suite:
```bash
python scripts/test_instagram_posting.py
```

3. Check status regularly:
```bash
python scripts/post_instagram.py --status
```

### Production Usage

1. Use the queue system:
```python
poster.add_to_queue("video.mp4")
poster.process_queue(max_posts=1)
```

2. Monitor analytics:
```python
status = poster.get_status()
success_rate = status['successful_posts'] / status['total_posts']
```

3. Handle errors gracefully:
```python
try:
    media_id = poster.upload_video("video.mp4")
    if not media_id:
        logger.error("Upload failed")
except Exception as e:
    logger.error(f"Error: {e}")
```

## Integration Examples

### With assemble_video.py

```python
from assemble_video import VideoAssembler
from post_instagram import InstagramPoster

# Generate video
assembler = VideoAssembler()
video_path = assembler.assemble_video(timeline_path, selections_path)

# Post to Instagram
poster = InstagramPoster()
media_id = poster.upload_video(video_path)
```

### With generate_script.py

```python
from generate_script import ScriptGenerator
from generate_caption import CaptionGenerator

# Generate script
generator = ScriptGenerator()
script = generator.generate_script(["AI", "technology"])

# Generate caption from script
caption_gen = CaptionGenerator()
caption = caption_gen.generate_from_script(script)
```

### Automated Workflow

```python
import schedule
import time

def daily_post():
    poster = InstagramPoster()
    poster.process_queue(max_posts=1)

# Schedule daily posts at 6 PM
schedule.every().day.at("18:00").do(daily_post)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Meta Graph API Reference

For advanced usage, refer to the official Meta documentation:

- [Instagram API Overview](https://developers.facebook.com/docs/instagram-api)
- [Content Publishing](https://developers.facebook.com/docs/instagram-api/guides/content-publishing)
- [Reels API](https://developers.facebook.com/docs/instagram-api/guides/reels)
- [Access Tokens](https://developers.facebook.com/docs/facebook-login/guides/access-tokens)

## Support

For issues or questions:

1. Check the logs for error messages
2. Run the test suite: `python scripts/test_instagram_posting.py`
3. Review the setup guide: `docs/INSTAGRAM_SETUP.md`
4. Check the quick start: `docs/INSTAGRAM_QUICK_START.md`
