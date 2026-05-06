.PHONY: install test run docker-build docker-up clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-logs:
	docker-compose logs -f

clean:
	rm -rf output/* logs/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available targets:"
	@echo "  install      - Install dependencies"
	@echo "  test        - Run tests"
	@echo "  run         - Run API locally"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-up   - Start API in Docker"
	@echo "  docker-logs - View Docker logs"
	@echo "  clean       - Clean temp files"