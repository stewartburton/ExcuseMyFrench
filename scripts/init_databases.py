#!/usr/bin/env python3
"""
Database Initialization Script for Excuse My French

This script creates and initializes SQLite databases for:
- Trends tracking (trending topics from various sources)
- Metrics storage (video performance data)
- Image library (character image metadata)

Usage:
    python scripts/init_databases.py [--reset] [--seed]

Options:
    --reset    Drop existing tables and recreate them
    --seed     Insert sample/initial data after creation
"""

import sqlite3
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv('config/.env')
except ImportError:
    print("Warning: python-dotenv not installed. Using default paths.")

# Database paths from environment or defaults
TRENDS_DB_PATH = os.getenv('TRENDS_DB_PATH', 'data/trends.db')
METRICS_DB_PATH = os.getenv('METRICS_DB_PATH', 'data/metrics.db')
IMAGE_LIBRARY_DB_PATH = os.getenv('IMAGE_LIBRARY_DB_PATH', 'data/image_library.db')

# Ensure paths are absolute
BASE_DIR = Path(__file__).parent.parent
TRENDS_DB_PATH = BASE_DIR / TRENDS_DB_PATH
METRICS_DB_PATH = BASE_DIR / METRICS_DB_PATH
IMAGE_LIBRARY_DB_PATH = BASE_DIR / IMAGE_LIBRARY_DB_PATH


class DatabaseInitializer:
    """Handles database creation and initialization."""

    def __init__(self, db_path: Path, db_name: str):
        self.db_path = db_path
        self.db_name = db_name
        self.conn: Optional[sqlite3.Connection] = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def connect(self):
        """Establish database connection."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        return self.conn

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.conn.close()

    def exists(self) -> bool:
        """Check if database file exists."""
        return self.db_path.exists()

    def execute_script(self, sql_script: str):
        """Execute a SQL script."""
        try:
            self.conn.executescript(sql_script)
            self.conn.commit()
            print(f"  ✓ Executed schema for {self.db_name}")
        except sqlite3.Error as e:
            print(f"  ✗ Error in {self.db_name}: {e}")
            raise

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return cursor.fetchone() is not None


def create_trends_schema() -> str:
    """Returns SQL schema for trends database."""
    return """
    -- Trending Topics Table
    CREATE TABLE IF NOT EXISTS trending_topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT NOT NULL,
        source TEXT NOT NULL,  -- e.g., 'google_trends', 'twitter', 'reddit'
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        relevance_score REAL DEFAULT 0.0,  -- 0.0 to 1.0 scale
        used_flag INTEGER DEFAULT 0,  -- 0 = unused, 1 = used
        category TEXT,  -- Optional categorization
        metadata TEXT,  -- JSON string for additional data
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_trending_topics_timestamp
        ON trending_topics(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_trending_topics_used_flag
        ON trending_topics(used_flag);

    CREATE INDEX IF NOT EXISTS idx_trending_topics_source
        ON trending_topics(source);

    CREATE INDEX IF NOT EXISTS idx_trending_topics_relevance
        ON trending_topics(relevance_score DESC);

    -- Combined index for common queries
    CREATE INDEX IF NOT EXISTS idx_trending_topics_unused_recent
        ON trending_topics(used_flag, timestamp DESC)
        WHERE used_flag = 0;

    -- Trigger to update updated_at timestamp
    CREATE TRIGGER IF NOT EXISTS update_trending_topics_timestamp
    AFTER UPDATE ON trending_topics
    BEGIN
        UPDATE trending_topics SET updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.id;
    END;

    -- Topic Search History (for tracking what we've searched for)
    CREATE TABLE IF NOT EXISTS search_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL,
        source TEXT NOT NULL,
        result_count INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_search_history_timestamp
        ON search_history(timestamp DESC);
    """


def create_metrics_schema() -> str:
    """Returns SQL schema for metrics database."""
    return """
    -- Video Performance Metrics Table
    CREATE TABLE IF NOT EXISTS video_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL,  -- Unique identifier for the video
        platform TEXT NOT NULL,  -- 'instagram', 'tiktok', 'youtube', etc.
        views INTEGER DEFAULT 0,
        likes INTEGER DEFAULT 0,
        comments INTEGER DEFAULT 0,
        shares INTEGER DEFAULT 0,
        saves INTEGER DEFAULT 0,  -- For Instagram
        watch_time_seconds INTEGER DEFAULT 0,
        engagement_rate REAL DEFAULT 0.0,  -- Calculated metric
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

        -- Ensure combination of video_id, platform, and timestamp is unique
        UNIQUE(video_id, platform, timestamp)
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_video_metrics_video_id
        ON video_metrics(video_id);

    CREATE INDEX IF NOT EXISTS idx_video_metrics_platform
        ON video_metrics(platform);

    CREATE INDEX IF NOT EXISTS idx_video_metrics_timestamp
        ON video_metrics(timestamp DESC);

    CREATE INDEX IF NOT EXISTS idx_video_metrics_engagement
        ON video_metrics(engagement_rate DESC);

    -- Video Metadata Table (stores additional info about videos)
    CREATE TABLE IF NOT EXISTS video_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id TEXT NOT NULL UNIQUE,
        title TEXT,
        description TEXT,
        topic TEXT,  -- The trending topic used
        script_path TEXT,
        audio_path TEXT,
        video_path TEXT,
        duration_seconds INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        posted_at DATETIME,
        status TEXT DEFAULT 'draft'  -- 'draft', 'rendered', 'posted', 'archived'
    );

    CREATE INDEX IF NOT EXISTS idx_video_metadata_status
        ON video_metadata(status);

    CREATE INDEX IF NOT EXISTS idx_video_metadata_topic
        ON video_metadata(topic);

    CREATE INDEX IF NOT EXISTS idx_video_metadata_created
        ON video_metadata(created_at DESC);

    -- Aggregate Metrics View (latest metrics per video)
    CREATE VIEW IF NOT EXISTS latest_video_metrics AS
    SELECT
        vm.*,
        vmd.title,
        vmd.topic,
        vmd.duration_seconds
    FROM video_metrics vm
    INNER JOIN (
        SELECT video_id, platform, MAX(timestamp) as max_timestamp
        FROM video_metrics
        GROUP BY video_id, platform
    ) latest ON vm.video_id = latest.video_id
        AND vm.platform = latest.platform
        AND vm.timestamp = latest.max_timestamp
    LEFT JOIN video_metadata vmd ON vm.video_id = vmd.video_id;
    """


def create_image_library_schema() -> str:
    """Returns SQL schema for image library database."""
    return """
    -- Image Library Table
    CREATE TABLE IF NOT EXISTS image_library (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        character TEXT NOT NULL,  -- 'butcher' or 'nutsy'
        emotion TEXT,  -- 'happy', 'sad', 'angry', 'neutral', 'excited', etc.
        pose TEXT,  -- 'sitting', 'standing', 'jumping', 'profile', etc.
        file_path TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,  -- 'generated', 'photo', 'dreambooth', 'stable_diffusion'
        file_size_bytes INTEGER,
        width INTEGER,
        height INTEGER,
        format TEXT,  -- 'png', 'jpg', 'webp'
        generation_prompt TEXT,  -- If AI-generated
        generation_model TEXT,  -- Model used for generation
        quality_score REAL,  -- 0.0 to 1.0, manual or auto-assessed
        usage_count INTEGER DEFAULT 0,  -- Track how many times used
        tags TEXT,  -- JSON array of tags
        metadata TEXT,  -- Additional JSON metadata
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_used_at DATETIME,

        -- Constraints
        CHECK (character IN ('butcher', 'nutsy'))
    );

    -- Indexes for performance
    CREATE INDEX IF NOT EXISTS idx_image_library_character
        ON image_library(character);

    CREATE INDEX IF NOT EXISTS idx_image_library_emotion
        ON image_library(emotion);

    CREATE INDEX IF NOT EXISTS idx_image_library_pose
        ON image_library(pose);

    CREATE INDEX IF NOT EXISTS idx_image_library_source
        ON image_library(source);

    CREATE INDEX IF NOT EXISTS idx_image_library_quality
        ON image_library(quality_score DESC);

    -- Combined index for common queries (find image by character and emotion)
    CREATE INDEX IF NOT EXISTS idx_image_library_char_emotion
        ON image_library(character, emotion);

    -- Image Usage Table (track which images were used in which videos)
    CREATE TABLE IF NOT EXISTS image_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER NOT NULL,
        video_id TEXT NOT NULL,
        scene_number INTEGER,  -- Which scene in the video
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY (image_id) REFERENCES image_library(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_image_usage_image_id
        ON image_usage(image_id);

    CREATE INDEX IF NOT EXISTS idx_image_usage_video_id
        ON image_usage(video_id);

    -- Trigger to update usage_count and last_used_at
    CREATE TRIGGER IF NOT EXISTS update_image_usage_stats
    AFTER INSERT ON image_usage
    BEGIN
        UPDATE image_library
        SET usage_count = usage_count + 1,
            last_used_at = CURRENT_TIMESTAMP
        WHERE id = NEW.image_id;
    END;

    -- Image Collections (group images for easy retrieval)
    CREATE TABLE IF NOT EXISTS image_collections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        description TEXT,
        character TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS collection_images (
        collection_id INTEGER NOT NULL,
        image_id INTEGER NOT NULL,
        sort_order INTEGER DEFAULT 0,

        PRIMARY KEY (collection_id, image_id),
        FOREIGN KEY (collection_id) REFERENCES image_collections(id) ON DELETE CASCADE,
        FOREIGN KEY (image_id) REFERENCES image_library(id) ON DELETE CASCADE
    );
    """


def seed_trends_data(db: DatabaseInitializer):
    """Insert sample data into trends database."""
    sample_topics = [
        ('AI trends 2025', 'google_trends', 0.95, 0, 'technology'),
        ('sustainable fashion', 'google_trends', 0.78, 0, 'lifestyle'),
        ('work from home tips', 'google_trends', 0.82, 0, 'productivity'),
    ]

    cursor = db.conn.cursor()
    cursor.executemany(
        """INSERT INTO trending_topics (topic, source, relevance_score, used_flag, category)
           VALUES (?, ?, ?, ?, ?)""",
        sample_topics
    )
    db.conn.commit()
    print(f"  ✓ Seeded {len(sample_topics)} sample topics")


def seed_metrics_data(db: DatabaseInitializer):
    """Insert sample data into metrics database."""
    # Sample video metadata
    sample_video = (
        'video_001',
        'Sample Video Title',
        'A funny conversation between Butcher and Nutsy',
        'AI trends 2025',
        None,
        None,
        'data/final_videos/video_001.mp4',
        45,
        'draft'
    )

    cursor = db.conn.cursor()
    cursor.execute(
        """INSERT INTO video_metadata
           (video_id, title, description, topic, script_path, audio_path, video_path, duration_seconds, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        sample_video
    )
    db.conn.commit()
    print(f"  ✓ Seeded sample video metadata")


def seed_image_library_data(db: DatabaseInitializer):
    """Insert sample data into image library database."""
    # Sample image collection
    cursor = db.conn.cursor()
    cursor.execute(
        """INSERT INTO image_collections (name, description, character)
           VALUES (?, ?, ?)""",
        ('Butcher Happy Collection', 'All happy poses of Butcher', 'butcher')
    )
    db.conn.commit()
    print(f"  ✓ Seeded sample image collection")


def init_database(db_path: Path, db_name: str, schema_func, seed_func=None, reset=False, seed=False):
    """Initialize a database with schema and optionally seed data."""
    print(f"\n{'='*60}")
    print(f"Initializing {db_name}")
    print(f"Path: {db_path}")
    print(f"{'='*60}")

    # Check if database exists
    db_existed = db_path.exists()
    if db_existed:
        print(f"  ℹ Database file exists: {db_path}")
    else:
        print(f"  ℹ Creating new database: {db_path}")

    with DatabaseInitializer(db_path, db_name) as db:
        # If reset flag is set, drop all tables
        if reset and db_existed:
            print("  ⚠ Resetting database (dropping all tables)...")
            cursor = db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                # Validate table name to prevent SQL injection
                if not table_name.replace('_', '').isalnum():
                    logger.warning(f"Skipping table with invalid name: {table_name}")
                    continue
                # Use parameterized query (not supported for table names, so validate instead)
                db.conn.execute(f"DROP TABLE IF EXISTS [{table_name}]")
            db.conn.commit()
            print("  ✓ All tables dropped")

        # Create schema
        print("  ⚙ Creating schema...")
        schema_sql = schema_func()
        db.execute_script(schema_sql)

        # Seed data if requested
        if seed and seed_func:
            print("  ⚙ Seeding initial data...")
            seed_func(db)

    print(f"  ✓ {db_name} initialization complete!")
    return True


def check_databases():
    """Check status of all databases."""
    print("\n" + "="*60)
    print("Database Status Check")
    print("="*60)

    databases = {
        'Trends Database': TRENDS_DB_PATH,
        'Metrics Database': METRICS_DB_PATH,
        'Image Library Database': IMAGE_LIBRARY_DB_PATH
    }

    for name, path in databases.items():
        if path.exists():
            size = path.stat().st_size
            print(f"  ✓ {name}: EXISTS ({size:,} bytes)")
        else:
            print(f"  ✗ {name}: NOT FOUND")

    print("="*60)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Initialize SQLite databases for Excuse My French project'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Drop existing tables and recreate them'
    )
    parser.add_argument(
        '--seed',
        action='store_true',
        help='Insert sample/initial data after creation'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Only check database status, do not create'
    )

    args = parser.parse_args()

    # Show initial status
    check_databases()

    if args.check:
        return

    # Confirmation for reset
    if args.reset:
        response = input("\n⚠  WARNING: This will delete all existing data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return

    print("\nStarting database initialization...\n")

    try:
        # Initialize each database
        init_database(
            TRENDS_DB_PATH,
            'Trends Database',
            create_trends_schema,
            seed_trends_data if args.seed else None,
            args.reset,
            args.seed
        )

        init_database(
            METRICS_DB_PATH,
            'Metrics Database',
            create_metrics_schema,
            seed_metrics_data if args.seed else None,
            args.reset,
            args.seed
        )

        init_database(
            IMAGE_LIBRARY_DB_PATH,
            'Image Library Database',
            create_image_library_schema,
            seed_image_library_data if args.seed else None,
            args.reset,
            args.seed
        )

        # Final status check
        check_databases()

        print("\n" + "="*60)
        print("All databases initialized successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Use these databases in your application")
        print("  2. Run with --seed flag to add sample data")
        print("  3. See schema details in this script")
        print("\nDatabase locations:")
        print(f"  - Trends: {TRENDS_DB_PATH}")
        print(f"  - Metrics: {METRICS_DB_PATH}")
        print(f"  - Image Library: {IMAGE_LIBRARY_DB_PATH}")
        print()

    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
