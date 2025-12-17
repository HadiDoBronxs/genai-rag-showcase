IMAGE_NAME = genai-rag-showcase

.PHONY: build run clean lint

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -p 8501:8501 -e OPENAI_API_KEY=$(OPENAI_API_KEY) $(IMAGE_NAME)

lint:
	# Running flake8 for code quality checks
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

clean:
	docker rmi $(IMAGE_NAME) || true
