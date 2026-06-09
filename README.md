# SPS GenAI Assignment 1

This project was developed using FastAPI and UV Package Manager.

## Features

### 1. Bigram Text Generation

Generate text based on a simple bigram language model.

Endpoint:

```http
POST /generate
```

Example Request:

```json
{
  "start_word": "bigram",
  "length": 10
}
```

### 2. Word Embedding API

Returns a spaCy word embedding vector using the en_core_web_md model.

Endpoint:

```http
POST /embedding
```

Example Request:

```json
{
  "word": "king"
}
```

## Run Locally

Install dependencies:

```bash
uv sync
```

Run FastAPI:

```bash
uv run fastapi dev main.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Docker Deployment

Build image:

```bash
docker build -t sps-genai .
```

Run container:

```bash
docker run -p 8000:80 sps-genai
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Project Structure

```text
sps_genai
│
├── app
│   ├── __init__.py
│   ├── bigram_model.py
│   └── embedding_model.py
│
├── main.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```