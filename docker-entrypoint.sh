#!/bin/bash
# Docker entrypoint script for ExcuseMyFrench
# This script performs runtime initialization that should not be baked into the image
set -e

echo "=== ExcuseMyFrench Docker Entrypoint ==="
echo "Running as user: $(whoami) (UID: $(id -u))"

# Function to check if database needs initialization
needs_db_init() {
    local db_path="$1"
    if [ ! -f "$db_path" ]; then
        return 0  # Database doesn't exist, needs init
    fi
    # Check if database has tables
    local table_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
    if [ "$table_count" -eq "0" ]; then
        return 0  # Database exists but has no tables, needs init
    fi
    return 1  # Database is initialized
}

# Initialize databases if they don't exist or are empty
echo "Checking database status..."
DB_INITIALIZED=false

if needs_db_init "data/trends.db" || needs_db_init "data/image_library.db" || needs_db_init "data/scripts.db"; then
    echo "Initializing databases..."
    python scripts/init_databases.py
    DB_INITIALIZED=true
else
    echo "Databases already initialized, skipping initialization"
fi

# Verify critical directories exist and are writable
echo "Verifying directory permissions..."
DIRS=(
    "data/scripts"
    "data/audio"
    "data/images"
    "data/videos"
    "data/animated"
    "data/final_videos"
    "data/instagram"
    "models/wan2.2"
    "models/sadtalker"
    "models/wav2lip"
)

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
    if [ ! -w "$dir" ]; then
        echo "WARNING: Directory not writable: $dir"
    fi
done

# Verify environment configuration
echo "Verifying environment configuration..."
if [ -f "scripts/validate_env.py" ]; then
    python scripts/validate_env.py || {
        echo "WARNING: Environment validation failed. Some features may not work."
        echo "Please check your config/.env file."
    }
else
    echo "Environment validation script not found, skipping..."
fi

echo "=== Initialization Complete ==="
echo "Starting application: $@"
echo ""

# Execute the main command (passed as arguments to this script)
exec "$@"
