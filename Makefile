.PHONY: install run test clean

install:
	pip install -r requirements.txt

run:
	uvicorn server:app --reload --port 8000

test:
	python -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
