# Instagram Posting Setup Guide

This guide will help you set up automated Instagram posting for your Excuse My French videos.

## Prerequisites

1. **Instagram Business Account**: You need a Business or Creator account
2. **Facebook Page**: Your Instagram account must be connected to a Facebook Page
3. **Meta Developer Account**: Required to create an app and get API access

## Step-by-Step Setup

### 1. Convert to Business Account

1. Open Instagram app on your phone
2. Go to Settings > Account > Switch to Professional Account
3. Choose "Business" or "Creator"
4. Complete the setup process

### 2. Create Facebook Page (if you don't have one)

1. Go to https://www.facebook.com/pages/create
2. Create a page for your Instagram content
3. Connect your Instagram Business Account to this page:
   - Facebook Page Settings > Instagram > Connect Account

### 3. Create Meta Developer App

1. Go to https://developers.facebook.com/
2. Click "My Apps" > "Create App"
3. Choose "Business" as the app type
4. Fill in app details:
   - App Name: "Excuse My French Poster" (or your choice)
   - Contact Email: Your email
5. Complete the security check

### 4. Configure Instagram API

1. In your app dashboard, click "Add Product"
2. Find "Instagram" and click "Set Up"
3. Go to Instagram > Basic Display > User Token Generator
4. Click "Generate Token" next to your Instagram account
5. Authorize the permissions:
   - instagram_basic
   - instagram_content_publish
   - pages_read_engagement
   - pages_show_list

### 5. Get Your Access Token

#### Option A: Short-Term Token (for testing)

1. Use the Graph API Explorer: https://developers.facebook.com/tools/explorer/
2. Select your app from the dropdown
3. Add permissions:
   - instagram_basic
   - instagram_content_publish
   - pages_read_engagement
4. Click "Generate Access Token"
5. This token expires in ~1 hour

#### Option B: Long-Lived Token (recommended)

1. Get a short-term token first (Option A)
2. Exchange it for a long-lived token using this command:

```bash
curl -i -X GET "https://graph.facebook.com/v21.0/oauth/access_token?grant_type=fb_exchange_token&client_id={your-app-id}&client_secret={your-app-secret}&fb_exchange_token={short-lived-token}"
```

3. This token lasts ~60 days and can be refreshed

### 6. Get Your Instagram User ID

Run this command with your access token:

```bash
curl -i -X GET "https://graph.facebook.com/v21.0/me/accounts?access_token={your-access-token}"
```

Find your Instagram account ID in the response. It looks like:
```json
{
  "id": "17841405309211844",
  "username": "your_username"
}
```

### 7. Configure Environment Variables

Edit your `config/.env` file and add:

```env
# Meta/Instagram API
META_ACCESS_TOKEN=your-long-lived-access-token-here
INSTAGRAM_USER_ID=your-instagram-user-id-here
INSTAGRAM_BUSINESS_ACCOUNT_ID=your-instagram-business-account-id-here

# Instagram Posting Settings
INSTAGRAM_POST_ENABLED=true
INSTAGRAM_HASHTAGS=#frenchbulldog #comedy #funny #reels #dogsofinstagram #fyp
```

### 8. Test the Setup

Test with a dry run:

```bash
python scripts/post_instagram.py data/final_videos/your_video.mp4 --dry-run
```

Check the status:

```bash
python scripts/post_instagram.py --status
```

## Important Notes

### Video Requirements

Instagram Reels have specific requirements:
- **Format**: MP4 or MOV
- **Resolution**: 1080x1920 (9:16 aspect ratio) - vertical format
- **Duration**: 3-90 seconds
- **File Size**: Max 100MB
- **Codec**: H.264 video, AAC audio

Your videos from `assemble_video.py` already meet these requirements!

### API Limitations

- **Rate Limits**:
  - 25 API calls per user per hour
  - 200 API calls per app per hour
- **Publishing Limits**: Instagram may limit how many posts you can make per day
- **Token Expiration**: Long-lived tokens expire after 60 days

### Best Practices

1. **Test First**: Always use `--dry-run` when testing
2. **Check Status**: Regularly monitor with `--status`
3. **Use Queue**: Schedule posts to avoid rate limits
4. **Monitor Analytics**: Check the analytics file for insights
5. **Refresh Tokens**: Set a reminder to refresh your access token monthly

## Troubleshooting

### "Access token is invalid"

- Your token may have expired
- Generate a new long-lived token
- Make sure you have the correct permissions

### "Rate limit exceeded"

- Wait an hour before posting again
- Use the queue system to spread out posts
- The script will automatically retry with exponential backoff

### "Video file too large"

- Check the file size (must be < 100MB)
- Reduce video quality or duration if needed
- Use FFmpeg to compress: `ffmpeg -i input.mp4 -b:v 2M output.mp4`

### "Invalid media type"

- Ensure your video is in MP4 or MOV format
- Check that it's vertical (1080x1920)
- Verify the codec is H.264

### "Publishing error"

- Check that your Instagram account is set to Business/Creator
- Verify the account is connected to a Facebook Page
- Ensure all required permissions are granted

## Advanced Features

### Caption Generation

The system can automatically generate engaging captions:

```bash
python scripts/generate_caption.py data/final_videos/video.mp4
```

### Queue Management

Add videos to a queue for later posting:

```bash
python scripts/post_instagram.py video.mp4 --queue
```

Process the queue (posts 1 video):

```bash
python scripts/post_instagram.py --process-queue
```

### Custom Captions

Override the auto-generated caption:

```bash
python scripts/post_instagram.py video.mp4 --caption "Custom caption here!"
```

### Scheduling

The queue system supports scheduled posting. Edit `queue.json` to set future post times:

```json
{
  "queue": [
    {
      "video_path": "video.mp4",
      "scheduled_time": "2024-01-01T18:00:00",
      "status": "pending"
    }
  ]
}
```

## Automation

### Daily Posting

Create a cron job (Linux/Mac) or Task Scheduler task (Windows) to auto-post daily:

**Linux/Mac (crontab):**
```bash
# Post one video from queue at 6 PM daily
0 18 * * * cd /path/to/ExcuseMyFrench && python scripts/post_instagram.py --process-queue
```

**Windows (Task Scheduler):**
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., Daily at 6 PM)
4. Action: Start a program
5. Program: `python`
6. Arguments: `scripts/post_instagram.py --process-queue`
7. Start in: `D:\ExcuseMyFrench\Repo\ExcuseMyFrench`

## Security

- **Never commit** your `.env` file to version control
- **Keep tokens secure**: Don't share them publicly
- **Rotate tokens regularly**: Generate new tokens every 30-60 days
- **Monitor usage**: Check the Meta Developer dashboard for suspicious activity

## Support Resources

- [Instagram API Documentation](https://developers.facebook.com/docs/instagram-api)
- [Meta Graph API Explorer](https://developers.facebook.com/tools/explorer/)
- [Access Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/)
- [Meta Developer Community](https://developers.facebook.com/community/)

## Next Steps

After successful setup:

1. Generate a video: `python scripts/assemble_video.py ...`
2. Generate caption: `python scripts/generate_caption.py video.mp4`
3. Post to Instagram: `python scripts/post_instagram.py video.mp4`
4. Check analytics: Review `data/instagram/analytics.json`

Happy posting!
