# n8n Orchestration Setup

This guide covers setting up n8n to automate the complete Excuse My French video generation pipeline.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Workflow Import](#workflow-import)
- [Configuration](#configuration)
- [Running Workflows](#running-workflows)
- [Monitoring](#monitoring)

---

## Overview

### What is n8n?

n8n is a fair-code workflow automation tool that connects different services and automates processes. For Excuse My French, it orchestrates the entire video generation pipeline from trending topics to final Instagram posts.

### Workflow Features

- **Scheduled Execution**: Daily automated video generation
- **Error Handling**: Graceful failure handling with notifications
- **Conditional Logic**: Skip steps if not needed (e.g., image generation)
- **Manual Override**: Ability to trigger pipeline manually
- **Progress Tracking**: Monitor each step's execution
- **Optional Auto-Posting**: Control whether to auto-post or review first

---

## Installation

### Option 1: Docker (Recommended)

**Prerequisites:**
- Docker Desktop installed
- 4GB+ RAM available
- Port 5678 available

**Installation Steps:**

1. **Create docker-compose.yml** in project root:
```yaml
version: '3.8'

services:
  n8n:
    image: n8nio/n8n:latest
    container_name: excusemyfrench-n8n
    restart: unless-stopped
    ports:
      - "5678:5678"
    environment:
      - N8N_BASIC_AUTH_ACTIVE=true
      - N8N_BASIC_AUTH_USER=admin
      - N8N_BASIC_AUTH_PASSWORD=changeme123  # CHANGE THIS!
      - N8N_HOST=localhost
      - N8N_PORT=5678
      - N8N_PROTOCOL=http
      - WEBHOOK_URL=http://localhost:5678/
      - GENERIC_TIMEZONE=America/New_York
    volumes:
      - ./n8n_data:/home/node/.n8n
      - ./workflows/n8n:/workflows
      - ./scripts:/scripts:ro
      - ./data:/data
      - ./excusemyfrench:/excusemyfrench:ro
    networks:
      - excusemyfrench-network

networks:
  excusemyfrench-network:
    driver: bridge
```

2. **Start n8n**:
```bash
docker-compose up -d
```

3. **Access n8n**:
- Open browser to http://localhost:5678
- Login with username: `admin`, password: `changeme123`
- **IMPORTANT**: Change the default password immediately

### Option 2: npm Installation

**Prerequisites:**
- Node.js 16+ installed
- npm or yarn package manager

**Installation Steps:**

1. **Install n8n globally**:
```bash
npm install -g n8n
```

2. **Start n8n**:
```bash
n8n start
```

3. **Access n8n**:
- Open browser to http://localhost:5678
- Create account on first visit

### Option 3: Desktop App

1. Download from https://n8n.io/download
2. Install and run
3. Access via built-in browser

---

## Workflow Import

### Import Main Pipeline Workflow

1. **Open n8n** (http://localhost:5678)

2. **Import workflow**:
   - Click **"Workflows"** in left sidebar
   - Click **"Import from File"**
   - Select `workflows/n8n/excuse_my_french_pipeline.json`
   - Click **"Import"**

3. **Review workflow**:
   - The workflow should appear in visual editor
   - Review each node and connection
   - Update paths if your project is in a different location

### Workflow Structure

```
Schedule Trigger (Daily at specified time)
  ↓
1. Fetch Trends
  ↓
2. Generate Script
  ↓
Parse Output
  ↓
3. Generate Audio
  ↓
4. Select Images
  ↓
Check Missing Images? → Yes → 5a. Generate Images → 5b. Re-select
  ↓ No                              ↓
Merge ←───────────────────────────┘
  ↓
6. Animate
  ↓
7. Assemble Video
  ↓
Check Auto Post? → Yes → 8. Post to Instagram
  ↓ No
Notify Manual Post Required
```

---

## Configuration

### Update Workflow Paths

If your project is not in `D:\ExcuseMyFrench\Repo\ExcuseMyFrench`, update paths in each node:

1. Click on each **"Execute Command"** node
2. Update the `cd` path to your project directory
3. Update the virtual environment activation path

**Example for Linux/Mac:**
```bash
cd /home/user/ExcuseMyFrench && source excusemyfrench/bin/activate && python scripts/fetch_trends.py
```

**Example for Windows:**
```bash
cd D:\ExcuseMyFrench\Repo\ExcuseMyFrench && excusemyfrench\Scripts\activate && python scripts/fetch_trends.py
```

### Configure Schedule

1. Click on **"Schedule Trigger"** node
2. Set your preferred schedule:
   - **Daily at specific time**: Set "Hours Interval" = 24, choose start time
   - **Twice daily**: Set "Hours Interval" = 12
   - **Weekly**: Set "Days of Week" = specific day
   - **Manual only**: Disable trigger, use manual execution

**Recommended Schedule:**
- Daily at 6:00 AM (generates video for same-day posting)
- Or Daily at 10:00 PM (generates overnight for next-day posting)

### Configure Auto-Posting

1. Open your `config/.env` file
2. Set `SKIP_POSTING` variable:
   ```bash
   SKIP_POSTING=false  # Auto-post to Instagram
   # or
   SKIP_POSTING=true   # Manual posting only
   ```

3. Save and restart n8n if running

### Add Error Notifications (Optional)

To receive notifications on errors:

1. **Add Email Node**:
   - Add new node after "Error Handler"
   - Choose "Email" node type
   - Configure SMTP settings
   - Set recipient email

2. **Add Slack Node** (alternative):
   - Add new node after "Error Handler"
   - Choose "Slack" node type
   - Configure Slack webhook URL
   - Set channel for notifications

---

## Running Workflows

### Manual Execution

To run the pipeline manually (testing or on-demand):

1. Open workflow in n8n
2. Click **"Execute Workflow"** button (top right)
3. Monitor progress in real-time
4. View output of each node
5. Check final video in `data/final_videos/`

### Scheduled Execution

Once configured, the workflow runs automatically:

1. n8n checks schedule trigger
2. At scheduled time, pipeline starts automatically
3. Executes each step sequentially
4. Logs all output
5. Sends notifications on completion/errors

### Execution History

View past executions:

1. Click **"Executions"** in left sidebar
2. See list of all workflow runs
3. Click any execution to view:
   - Start/end time
   - Duration
   - Output of each step
   - Any errors encountered

---

## Monitoring

### Real-Time Monitoring

Monitor active workflow execution:

1. Open workflow
2. Click **"Execute Workflow"**
3. Watch nodes turn green as they complete
4. Click any node to see its output
5. View error details in red nodes (if any)

### Execution Logs

Access detailed logs:

1. **n8n Logs**:
   - Docker: `docker logs excusemyfrench-n8n`
   - npm: Check terminal where n8n is running

2. **Pipeline Logs**:
   - Each script logs to `data/logs/app.log`
   - View with: `tail -f data/logs/app.log`

### Performance Metrics

Track pipeline performance:

1. **Execution Time**: View in "Executions" list
2. **Success Rate**: Count successful vs failed executions
3. **Bottlenecks**: Identify slowest steps

**Typical Execution Times:**
- Fetch Trends: 5-10 seconds
- Generate Script: 10-30 seconds (LLM dependent)
- Generate Audio: 1-2 minutes (depends on script length)
- Select Images: 5-10 seconds
- Generate Images: 5-15 minutes (if needed, GPU dependent)
- Animate: 10-30 minutes (method and GPU dependent)
- Assemble Video: 1-3 minutes
- **Total**: 20-50 minutes (without image generation)

---

## Troubleshooting

### Workflow Fails to Start

**Issue**: Schedule trigger doesn't fire
- **Solution**: Check n8n is running (`docker ps` or check terminal)
- **Solution**: Verify trigger is activated (should be blue/green)
- **Solution**: Check system time is correct

### Command Execution Errors

**Issue**: "Command not found" or "Python not found"
- **Solution**: Update paths in Execute Command nodes
- **Solution**: Verify virtual environment is activated in command
- **Solution**: Test commands manually in terminal first

### Path Issues

**Issue**: "File not found" or "No such file or directory"
- **Solution**: Use absolute paths in all commands
- **Solution**: Check Windows vs Linux path formats (\ vs /)
- **Solution**: Verify project directory structure

### Permission Errors

**Issue**: "Permission denied" when executing scripts
- **Solution**: Make scripts executable: `chmod +x scripts/*.py`
- **Solution**: Check Docker volume permissions
- **Solution**: Run n8n with appropriate user permissions

### Memory Issues

**Issue**: n8n container crashes or freezes
- **Solution**: Increase Docker memory allocation (Settings → Resources)
- **Solution**: Reduce concurrent executions
- **Solution**: Optimize pipeline scripts to use less memory

---

## Advanced Configuration

### Parallel Execution

For faster processing, run independent steps in parallel:

1. Duplicate workflow
2. Split into parallel branches:
   - Branch A: Script → Audio
   - Branch B: Trends → (process separately)
3. Merge results before animation step

### Webhook Triggers

Trigger pipeline via webhook (API call):

1. Replace "Schedule Trigger" with "Webhook" node
2. Configure webhook URL
3. Call from external service:
   ```bash
   curl -X POST http://localhost:5678/webhook/generate-video
   ```

### Multi-Environment Setup

Run different configs for testing vs production:

1. Create separate workflows:
   - `excuse_my_french_pipeline_prod.json`
   - `excuse_my_french_pipeline_test.json`
2. Use different .env files
3. Output to different directories

---

## Environment Variables

n8n can access environment variables from .env:

```bash
# n8n will read these automatically when using Docker
SKIP_POSTING=false
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...
```

Access in workflow nodes:
```
{{ $env.SKIP_POSTING }}
{{ $env.OPENAI_API_KEY }}
```

---

## Backup and Restore

### Backup Workflows

1. **Export workflow**:
   - Open workflow
   - Click "..." menu → Download
   - Save JSON file

2. **Backup n8n data** (Docker):
   ```bash
   docker exec excusemyfrench-n8n n8n export:workflow --backup --output=/home/node/.n8n/backups/
   ```

### Restore Workflows

1. **Import workflow**:
   - Click "Workflows" → "Import from File"
   - Select saved JSON file

2. **Restore n8n data** (Docker):
   ```bash
   docker exec excusemyfrench-n8n n8n import:workflow --input=/home/node/.n8n/backups/workflow.json
   ```

---

## Next Steps

1. Install n8n (Docker recommended)
2. Import main pipeline workflow
3. Update paths for your system
4. Configure schedule trigger
5. Test manual execution
6. Review execution logs
7. Set up error notifications
8. Enable scheduled execution

For more information:
- n8n Documentation: https://docs.n8n.io
- n8n Community: https://community.n8n.io
