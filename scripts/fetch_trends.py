#!/usr/bin/env python3
"""
Fetch trending topics from Google Trends and store them in SQLite database.

This script uses pytrends to query Google Trends for trending searches and
stores them in a local database for later use in script generation.
"""

import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from pytrends.request import TrendReq

# Load environment variables
load_dotenv(Path(__file__).parent.parent / "config" / ".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrendsFetcher:
    """Fetches and stores Google Trends data."""

    def __init__(self, db_path: str = "data/trends.db"):
        """
        Initialize the TrendsFetcher.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self._init_database()

    def _init_database(self):
        """Initialize the database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    keyword TEXT NOT NULL,
                    region TEXT NOT NULL,
                    interest INTEGER,
                    category TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    fetched_date DATE,
                    UNIQUE(keyword, region, fetched_date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_keyword ON trends(keyword)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON trends(timestamp DESC)
            """)
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    def fetch_trending_searches(self, region: str = "united_states") -> List[str]:
        """
        Fetch current trending searches from Google Trends.

        Args:
            region: The region to fetch trends for (default: united_states)

        Returns:
            List of trending search terms
        """
        try:
            logger.info(f"Fetching trending searches for {region}...")
            trending_searches_df = self.pytrends.trending_searches(pn=region)

            if trending_searches_df is not None and not trending_searches_df.empty:
                trends = trending_searches_df[0].tolist()
                logger.info(f"Found {len(trends)} trending searches")
                return trends
            else:
                logger.warning("No trending searches found")
                return []

        except Exception as e:
            logger.error(f"Error fetching trending searches: {e}")
            return []

    def fetch_interest_over_time(
        self,
        keywords: List[str],
        timeframe: str = "now 7-d"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch interest over time for given keywords.

        Args:
            keywords: List of keywords to check
            timeframe: Timeframe for the data (e.g., "now 7-d", "today 1-m")

        Returns:
            DataFrame with interest over time data or None if error
        """
        try:
            logger.info(f"Fetching interest over time for {len(keywords)} keywords...")
            self.pytrends.build_payload(keywords, timeframe=timeframe)
            interest_df = self.pytrends.interest_over_time()

            if interest_df is not None and not interest_df.empty:
                # Remove the 'isPartial' column if it exists
                if 'isPartial' in interest_df.columns:
                    interest_df = interest_df.drop(columns=['isPartial'])
                logger.info(f"Retrieved interest data for {len(interest_df)} time points")
                return interest_df
            else:
                logger.warning("No interest data found")
                return None

        except Exception as e:
            logger.error(f"Error fetching interest over time: {e}")
            return None

    def fetch_related_queries(self, keyword: str) -> dict:
        """
        Fetch related queries for a given keyword.

        Args:
            keyword: The keyword to find related queries for

        Returns:
            Dictionary containing 'top' and 'rising' related queries
        """
        try:
            logger.info(f"Fetching related queries for '{keyword}'...")
            self.pytrends.build_payload([keyword])
            related = self.pytrends.related_queries()

            if keyword in related:
                return related[keyword]
            else:
                logger.warning(f"No related queries found for '{keyword}'")
                return {'top': None, 'rising': None}

        except Exception as e:
            logger.error(f"Error fetching related queries: {e}")
            return {'top': None, 'rising': None}

    def store_trends(
        self,
        trends: List[str],
        region: str = "united_states",
        category: str = "general"
    ):
        """
        Store trending keywords in the database.

        Args:
            trends: List of trending keywords
            region: The region these trends are from
            category: Category of trends (e.g., 'general', 'entertainment')
        """
        if not trends:
            logger.warning("No trends to store")
            return

        today = datetime.now().date()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stored_count = 0

            for keyword in trends:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO trends
                        (keyword, region, interest, category, fetched_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (keyword, region, 100, category, today))

                    if cursor.rowcount > 0:
                        stored_count += 1

                except sqlite3.Error as e:
                    logger.error(f"Error storing keyword '{keyword}': {e}")

            conn.commit()
            logger.info(f"Stored {stored_count} new trends in database")

    def store_interest_data(
        self,
        interest_df: pd.DataFrame,
        region: str = "united_states"
    ):
        """
        Store interest over time data in the database.

        Args:
            interest_df: DataFrame with interest data
            region: The region this data is from
        """
        if interest_df is None or interest_df.empty:
            logger.warning("No interest data to store")
            return

        today = datetime.now().date()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            stored_count = 0

            for keyword in interest_df.columns:
                # Get the average interest for this keyword
                avg_interest = int(interest_df[keyword].mean())

                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO trends
                        (keyword, region, interest, category, fetched_date)
                        VALUES (?, ?, ?, ?, ?)
                    """, (keyword, region, avg_interest, "interest_over_time", today))
                    stored_count += 1

                except sqlite3.Error as e:
                    logger.error(f"Error storing interest data for '{keyword}': {e}")

            conn.commit()
            logger.info(f"Stored interest data for {stored_count} keywords")

    def get_recent_trends(self, days: int = 7, limit: int = 20) -> List[dict]:
        """
        Retrieve recent trends from the database.

        Args:
            days: Number of days to look back
            limit: Maximum number of trends to return

        Returns:
            List of trend dictionaries
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT keyword, region, interest, category, timestamp
                FROM trends
                WHERE timestamp >= ?
                ORDER BY interest DESC, timestamp DESC
                LIMIT ?
            """, (cutoff_date, limit))

            rows = cursor.fetchall()
            trends = [
                {
                    'keyword': row[0],
                    'region': row[1],
                    'interest': row[2],
                    'category': row[3],
                    'timestamp': row[4]
                }
                for row in rows
            ]

            logger.info(f"Retrieved {len(trends)} trends from last {days} days")
            return trends


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Fetch Google Trends data and store in database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch current trending searches for US
  python fetch_trends.py

  # Fetch trends for a specific region
  python fetch_trends.py --region india

  # Fetch interest data for specific keywords
  python fetch_trends.py --keywords "artificial intelligence" "machine learning"

  # Display recent trends from database
  python fetch_trends.py --show-recent --days 7 --limit 10
        """
    )

    parser.add_argument(
        "--region",
        default="united_states",
        help="Region to fetch trends for (default: united_states)"
    )

    parser.add_argument(
        "--keywords",
        nargs="+",
        help="Specific keywords to fetch interest data for"
    )

    parser.add_argument(
        "--timeframe",
        default="now 7-d",
        help="Timeframe for interest data (default: now 7-d)"
    )

    parser.add_argument(
        "--db-path",
        default="data/trends.db",
        help="Path to trends database (default: data/trends.db)"
    )

    parser.add_argument(
        "--show-recent",
        action="store_true",
        help="Display recent trends from database"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days to look back for recent trends (default: 7)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of trends to retrieve (default: 20)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Initialize fetcher
    fetcher = TrendsFetcher(db_path=args.db_path)

    if args.show_recent:
        # Display recent trends
        trends = fetcher.get_recent_trends(days=args.days, limit=args.limit)

        if trends:
            print(f"\nRecent trends (last {args.days} days):")
            print("-" * 80)
            for i, trend in enumerate(trends, 1):
                print(f"{i:2d}. {trend['keyword']:40s} "
                      f"Interest: {trend['interest']:3d} "
                      f"Region: {trend['region']:15s} "
                      f"({trend['timestamp']})")
        else:
            print("No recent trends found in database")

    elif args.keywords:
        # Fetch interest data for specific keywords
        # Process in batches of 5 (Google Trends API limit)
        batch_size = 5
        for i in range(0, len(args.keywords), batch_size):
            batch = args.keywords[i:i+batch_size]
            interest_df = fetcher.fetch_interest_over_time(batch, args.timeframe)

            if interest_df is not None:
                fetcher.store_interest_data(interest_df, args.region)
                print(f"\nInterest data for: {', '.join(batch)}")
                print(interest_df.tail())

    else:
        # Fetch current trending searches
        trends = fetcher.fetch_trending_searches(region=args.region)

        if trends:
            fetcher.store_trends(trends, region=args.region)
            print(f"\nFetched {len(trends)} trending searches for {args.region}:")
            print("-" * 80)
            for i, trend in enumerate(trends, 1):
                print(f"{i:2d}. {trend}")
        else:
            logger.error("Failed to fetch trends")
            sys.exit(1)


if __name__ == "__main__":
    main()
