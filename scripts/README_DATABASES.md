# Database Initialization Guide

This guide explains how to initialize and manage the SQLite databases for the Excuse My French project.

## Overview

The project uses three SQLite databases to manage different aspects of the video generation pipeline:

1. **trends.db** - Stores trending topics and search history
2. **metrics.db** - Tracks video performance metrics across platforms
3. **image_library.db** - Manages character images and their metadata

## Quick Start

### Initialize All Databases

```bash
python scripts/init_databases.py
```

This will create all three databases with their complete schemas in the `data/` directory.

### Check Database Status

```bash
python scripts/init_databases.py --check
```

### Initialize with Sample Data

```bash
python scripts/init_databases.py --seed
```

### Reset and Recreate Databases

```bash
python scripts/init_databases.py --reset
```

**Warning:** This will delete all existing data!

## Database Schemas

### 1. Trends Database (trends.db)

#### Tables

**trending_topics**
- `id` - Primary key (auto-increment)
- `topic` - The trending topic text
- `source` - Source of the trend (e.g., 'google_trends', 'twitter', 'reddit')
- `timestamp` - When the trend was captured
- `relevance_score` - Relevance score (0.0 to 1.0)
- `used_flag` - Whether this topic has been used (0 = unused, 1 = used)
- `category` - Optional category (e.g., 'technology', 'lifestyle')
- `metadata` - JSON string for additional data
- `created_at` - Record creation timestamp
- `updated_at` - Last update timestamp (auto-updated via trigger)

**search_history**
- `id` - Primary key (auto-increment)
- `query` - Search query text
- `source` - Source platform
- `result_count` - Number of results found
- `timestamp` - When the search was performed

#### Indexes
- `idx_trending_topics_timestamp` - Fast queries by date
- `idx_trending_topics_used_flag` - Quick filtering of unused topics
- `idx_trending_topics_source` - Filter by source
- `idx_trending_topics_relevance` - Sort by relevance
- `idx_trending_topics_unused_recent` - Combined index for unused recent topics

#### Triggers
- `update_trending_topics_timestamp` - Auto-updates `updated_at` field on record changes

---

### 2. Metrics Database (metrics.db)

#### Tables

**video_metrics**
- `id` - Primary key (auto-increment)
- `video_id` - Unique video identifier
- `platform` - Platform name ('instagram', 'tiktok', 'youtube', etc.)
- `views` - View count
- `likes` - Like count
- `comments` - Comment count
- `shares` - Share count
- `saves` - Save count (Instagram-specific)
- `watch_time_seconds` - Total watch time
- `engagement_rate` - Calculated engagement rate (0.0 to 1.0)
- `timestamp` - Metrics snapshot time
- `created_at` - Record creation timestamp
- **Unique constraint:** (video_id, platform, timestamp)

**video_metadata**
- `id` - Primary key (auto-increment)
- `video_id` - Unique video identifier
- `title` - Video title
- `description` - Video description
- `topic` - The trending topic used
- `script_path` - Path to the script file
- `audio_path` - Path to the audio file
- `video_path` - Path to the final video file
- `duration_seconds` - Video duration
- `created_at` - Record creation timestamp
- `posted_at` - When video was posted
- `status` - Current status ('draft', 'rendered', 'posted', 'archived')

#### Views

**latest_video_metrics**
- Combines latest metrics with video metadata
- Shows most recent metrics for each video on each platform

#### Indexes
- `idx_video_metrics_video_id` - Fast video lookup
- `idx_video_metrics_platform` - Filter by platform
- `idx_video_metrics_timestamp` - Sort by date
- `idx_video_metrics_engagement` - Sort by engagement
- `idx_video_metadata_status` - Filter by status
- `idx_video_metadata_topic` - Filter by topic
- `idx_video_metadata_created` - Sort by creation date

---

### 3. Image Library Database (image_library.db)

#### Tables

**image_library**
- `id` - Primary key (auto-increment)
- `character` - Character name ('butcher' or 'nutsy') - constrained via CHECK
- `emotion` - Emotion state (e.g., 'happy', 'sad', 'angry', 'neutral', 'excited')
- `pose` - Pose description (e.g., 'sitting', 'standing', 'jumping', 'profile')
- `file_path` - Unique path to the image file
- `source` - Image source ('generated', 'photo', 'dreambooth', 'stable_diffusion')
- `file_size_bytes` - File size in bytes
- `width` - Image width in pixels
- `height` - Image height in pixels
- `format` - File format ('png', 'jpg', 'webp')
- `generation_prompt` - Prompt used for AI generation (if applicable)
- `generation_model` - Model used for generation
- `quality_score` - Quality assessment (0.0 to 1.0)
- `usage_count` - Number of times used (auto-incremented via trigger)
- `tags` - JSON array of tags
- `metadata` - Additional JSON metadata
- `timestamp` - Image capture/generation time
- `created_at` - Record creation timestamp
- `last_used_at` - Last usage timestamp (auto-updated via trigger)

**image_usage**
- `id` - Primary key (auto-increment)
- `image_id` - Foreign key to image_library
- `video_id` - Video identifier where image was used
- `scene_number` - Scene number in the video
- `timestamp` - When the image was used
- **Foreign key:** References image_library(id) with CASCADE delete

**image_collections**
- `id` - Primary key (auto-increment)
- `name` - Unique collection name
- `description` - Collection description
- `character` - Character filter (optional)
- `created_at` - Record creation timestamp

**collection_images**
- `collection_id` - Foreign key to image_collections
- `image_id` - Foreign key to image_library
- `sort_order` - Display order in collection
- **Composite primary key:** (collection_id, image_id)
- **Foreign keys:** Both with CASCADE delete

#### Indexes
- `idx_image_library_character` - Filter by character
- `idx_image_library_emotion` - Filter by emotion
- `idx_image_library_pose` - Filter by pose
- `idx_image_library_source` - Filter by source
- `idx_image_library_quality` - Sort by quality
- `idx_image_library_char_emotion` - Combined index for character + emotion queries
- `idx_image_usage_image_id` - Fast image usage lookup
- `idx_image_usage_video_id` - Fast video usage lookup

#### Triggers
- `update_image_usage_stats` - Auto-increments `usage_count` and updates `last_used_at` when image is used

---

## Script Features

### Idempotent Design
The script is safe to run multiple times. It uses `CREATE TABLE IF NOT EXISTS` to avoid errors on subsequent runs.

### WAL Mode
All databases are initialized with Write-Ahead Logging (WAL) mode for better concurrency and performance.

### Foreign Key Constraints
Foreign key constraints are enabled for referential integrity in the image library database.

### Auto-Updating Timestamps
Triggers automatically maintain `updated_at` and `last_used_at` fields.

## Usage Examples

### Python Code Example

```python
import sqlite3
from pathlib import Path

# Connect to trends database
trends_db = Path('data/trends.db')
conn = sqlite3.connect(trends_db)

# Insert a trending topic
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO trending_topics (topic, source, relevance_score, category)
    VALUES (?, ?, ?, ?)
""", ('AI art generators', 'google_trends', 0.92, 'technology'))
conn.commit()

# Query unused topics
cursor.execute("""
    SELECT topic, relevance_score
    FROM trending_topics
    WHERE used_flag = 0
    ORDER BY relevance_score DESC
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"Topic: {row[0]}, Score: {row[1]}")

conn.close()
```

### SQLite Command Line

```bash
# Open database
sqlite3 data/trends.db

# View tables
.tables

# View schema
.schema trending_topics

# Query data
SELECT * FROM trending_topics WHERE used_flag = 0;

# Exit
.quit
```

## Future Schema Updates

When updating the schema:

1. **Add migration scripts** - Create versioned migration files for schema changes
2. **Backup first** - Always backup databases before applying migrations
3. **Test on copy** - Test migrations on a copy of production data
4. **Use ALTER TABLE** - For adding columns or indexes to existing tables
5. **Avoid breaking changes** - Ensure backward compatibility when possible

### Example Migration Pattern

```python
def migrate_v1_to_v2(db_path):
    """Add new column to trending_topics table."""
    conn = sqlite3.connect(db_path)
    try:
        # Add new column
        conn.execute("""
            ALTER TABLE trending_topics
            ADD COLUMN priority INTEGER DEFAULT 0
        """)
        # Add index
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trending_topics_priority
            ON trending_topics(priority DESC)
        """)
        conn.commit()
        print("Migration successful!")
    except sqlite3.Error as e:
        print(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()
```

## Database Locations

By default, databases are stored in:
- `data/trends.db`
- `data/metrics.db`
- `data/image_library.db`

These paths are configured in `config/.env` and can be customized:

```bash
TRENDS_DB_PATH=data/trends.db
METRICS_DB_PATH=data/metrics.db
IMAGE_LIBRARY_DB_PATH=data/image_library.db
```

## Backup Recommendations

### Manual Backup
```bash
# Backup all databases
cp data/trends.db data/backups/trends_backup_$(date +%Y%m%d).db
cp data/metrics.db data/backups/metrics_backup_$(date +%Y%m%d).db
cp data/image_library.db data/backups/image_library_backup_$(date +%Y%m%d).db
```

### Automated Backup Script
Create a cron job or scheduled task to run backups regularly.

## Troubleshooting

### Database is locked
- Ensure no other process is accessing the database
- Check for lingering connections in your code
- WAL mode helps reduce lock contention

### Foreign key constraint failed
- Ensure foreign key constraints are enabled: `PRAGMA foreign_keys = ON`
- Verify referenced records exist before inserting

### Performance issues
- Use EXPLAIN QUERY PLAN to analyze slow queries
- Ensure appropriate indexes exist
- Consider VACUUM to optimize database file

## Support

For issues or questions:
1. Check the main README.md
2. Review the script source code: `scripts/init_databases.py`
3. Open an issue on the project repository

---

Generated by init_databases.py
