.PHONY: help install setup test clean run-pipeline lint format check-env

# Default target
help:
	@echo "ExcuseMyFrench - Makefile Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make setup            - Initial setup (databases, directories)"
	@echo "  make check-env        - Verify environment configuration"
	@echo ""
	@echo "Pipeline Operations:"
	@echo "  make run-pipeline     - Run the complete video generation pipeline"
	@echo "  make fetch-trends     - Fetch trending topics"
	@echo "  make generate-script  - Generate script from trends"
	@echo "  make generate-audio   - Generate audio from latest script"
	@echo "  make assemble-video   - Assemble final video"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test             - Run all tests"
	@echo "  make test-comfyui     - Test ComfyUI integration"
	@echo "  make test-instagram   - Test Instagram posting (dry-run)"
	@echo "  make lint             - Run code linters"
	@echo "  make format           - Format code with black"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean            - Clean temporary files and caches"
	@echo "  make clean-all        - Clean everything including data"
	@echo "  make backup-db        - Backup all databases"
	@echo "  make stats            - Show project statistics"
	@echo ""
	@echo "ComfyUI:"
	@echo "  make setup-comfyui    - Install and configure ComfyUI"
	@echo "  make start-comfyui    - Start ComfyUI server"
	@echo ""
	@echo "Training:"
	@echo "  make train-butcher    - Train DreamBooth model for Butcher"
	@echo "  make prepare-training - Prepare training data"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Installation complete!"

# Setup
setup: install
	@echo "Setting up ExcuseMyFrench..."
	@echo "Creating directories..."
	@mkdir -p data/scripts data/audio data/images data/videos data/animated data/final_videos
	@mkdir -p data/instagram models/wan2.2 models/sadtalker models/wav2lip
	@echo "Initializing databases..."
	python scripts/init_databases.py
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "1. Copy config/.env.example to config/.env"
	@echo "2. Add your API keys to config/.env"
	@echo "3. Run 'make check-env' to verify configuration"

check-env:
	@echo "Checking environment configuration..."
	@python -c "from dotenv import load_dotenv; import os; from pathlib import Path; \
		load_dotenv(Path('config/.env')); \
		required = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']; \
		missing = [k for k in required if not os.getenv(k)]; \
		print('✓ All required API keys found!' if not missing else f'✗ Missing: {missing}')"

# Pipeline Operations
run-pipeline:
	@echo "Running complete pipeline..."
	python scripts/run_pipeline.py

fetch-trends:
	@echo "Fetching trending topics..."
	python scripts/fetch_trends.py --days 7 --limit 10

generate-script:
	@echo "Generating script from latest trends..."
	python scripts/generate_script.py --from-trends --days 3

generate-audio:
	@echo "Generating audio (requires script file)..."
	@echo "Usage: make generate-audio SCRIPT=path/to/script.json"
	@test -n "$(SCRIPT)" || (echo "Error: Please specify SCRIPT=path/to/script.json" && exit 1)
	python scripts/generate_audio.py "$(SCRIPT)"

assemble-video:
	@echo "Assembling video (requires timeline and images)..."
	@echo "Usage: make assemble-video TIMELINE=path IMAGES=path"
	@test -n "$(TIMELINE)" || (echo "Error: Please specify TIMELINE=path" && exit 1)
	@test -n "$(IMAGES)" || (echo "Error: Please specify IMAGES=path" && exit 1)
	python scripts/assemble_video.py --timeline "$(TIMELINE)" --images "$(IMAGES)"

# Testing
test:
	@echo "Running tests..."
	@python -m pytest tests/ -v || echo "Note: Install pytest to run tests (pip install pytest)"

test-comfyui:
	@echo "Testing ComfyUI integration..."
	python scripts/test_comfyui.py

test-instagram:
	@echo "Testing Instagram posting (dry-run)..."
	python scripts/test_instagram_posting.py

# Code Quality
lint:
	@echo "Running linters..."
	@python -m flake8 scripts/ --max-line-length=120 --exclude=__pycache__ || echo "Note: Install flake8 to run linter (pip install flake8)"
	@python -m pylint scripts/*.py --max-line-length=120 || echo "Note: Install pylint to run linter (pip install pylint)"

format:
	@echo "Formatting code with black..."
	@python -m black scripts/ --line-length=120 || echo "Note: Install black to format code (pip install black)"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*~" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"

clean-all: clean
	@echo "WARNING: This will delete all generated data!"
	@echo "Press Ctrl+C to cancel, or Enter to continue..."
	@read confirm
	rm -rf data/scripts/* data/audio/* data/images/* data/videos/* data/animated/* data/final_videos/*
	@echo "All data cleaned!"

# Backup
backup-db:
	@echo "Backing up databases..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	cp data/*.db backups/backup_$${timestamp}/ 2>/dev/null || echo "No databases to backup"; \
	echo "Backup complete: backups/backup_$${timestamp}/"

# Statistics
stats:
	@echo "Project Statistics"
	@echo "=================="
	@echo ""
	@echo "Scripts:"
	@find scripts -name "*.py" | wc -l | xargs echo "  Python files:"
	@find scripts -name "*.py" -exec wc -l {} + | tail -1 | awk '{print "  Total lines: " $$1}'
	@echo ""
	@echo "Generated Content:"
	@find data/scripts -name "*.json" 2>/dev/null | wc -l | xargs echo "  Scripts:"
	@find data/audio -name "*.mp3" 2>/dev/null | wc -l | xargs echo "  Audio files:"
	@find data/final_videos -name "*.mp4" 2>/dev/null | wc -l | xargs echo "  Videos:"
	@echo ""
	@echo "Database Records:"
	@python -c "import sqlite3; \
		try: \
			conn = sqlite3.connect('data/content.db'); \
			cursor = conn.cursor(); \
			cursor.execute('SELECT COUNT(*) FROM trending_topics'); \
			print(f'  Trending topics: {cursor.fetchone()[0]}'); \
			cursor.execute('SELECT COUNT(*) FROM generated_scripts'); \
			print(f'  Scripts: {cursor.fetchone()[0]}'); \
			conn.close(); \
		except: print('  Database not initialized')" 2>/dev/null || echo "  Database not initialized"

# ComfyUI
setup-comfyui:
	@echo "Setting up ComfyUI..."
	python scripts/setup_comfyui.py

start-comfyui:
	@echo "Starting ComfyUI server..."
	@echo "Note: Make sure ComfyUI is installed first (make setup-comfyui)"
	@COMFYUI_PATH=$$(python -c "import os; from dotenv import load_dotenv; from pathlib import Path; \
		load_dotenv(Path('config/.env')); print(os.getenv('COMFYUI_PATH', 'ComfyUI'))"); \
	cd $$COMFYUI_PATH && python main.py

# Training
prepare-training:
	@echo "Preparing training data..."
	python scripts/prepare_training_data.py

train-butcher:
	@echo "Training DreamBooth model for Butcher..."
	@echo "This may take several hours..."
	python scripts/train_dreambooth.py --config training/config/butcher_config.yaml

# Docker targets (if Docker is available)
docker-build:
	@echo "Building Docker image..."
	docker build -t excusemyfrench:latest .

docker-run:
	@echo "Running in Docker container..."
	docker run -it --gpus all -v $$(pwd)/data:/app/data excusemyfrench:latest

# Development
dev-install: install
	@echo "Installing development dependencies..."
	pip install black flake8 pylint pytest pytest-cov ipython
	@echo "Development environment ready!"

watch:
	@echo "Watching for changes..."
	@echo "Note: Install watchdog to use this feature (pip install watchdog)"
	@python -c "from watchdog.observers import Observer; from watchdog.events import FileSystemEventHandler; \
		import time; \
		class Handler(FileSystemEventHandler): \
			def on_modified(self, event): print(f'Modified: {event.src_path}'); \
		observer = Observer(); observer.schedule(Handler(), 'scripts', recursive=True); \
		observer.start(); \
		try: \
			while True: time.sleep(1); \
		except KeyboardInterrupt: observer.stop(); \
		observer.join()"
