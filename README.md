# SPS GenAI Assignments

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

---

### 2. Word Embedding API

Returns a spaCy word embedding vector using the `en_core_web_md` model.

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

---

### 3. CNN Image Classification API

This assignment extends the project by implementing a Convolutional Neural Network (CNN) trained on the CIFAR10 dataset.

Features:

* CNN image classifier
* CIFAR10 dataset training
* Train/validation split
* Checkpoint saving and loading
* FastAPI image classification endpoint
* Docker deployment

Endpoint:

```http
POST /predict-image
```

Example Response:

```json
{
  "filename": "plane.jpg",
  "predicted_class": "airplane",
  "confidence": 0.89
}
```

### CNN Architecture

Input: 64 × 64 × 3 RGB image

Layers:

1. Conv2D (16 filters, 3×3 kernel)
2. ReLU
3. MaxPooling2D
4. Conv2D (32 filters, 3×3 kernel)
5. ReLU
6. MaxPooling2D
7. Flatten
8. Fully Connected Layer (100 units)
9. ReLU
10. Fully Connected Layer (10 output classes)

### CIFAR10 Classes

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

### Training

Run:

```bash
uv run python train_cnn.py
```

The training pipeline includes:

* Data loading and preprocessing
* Train/validation split
* Model checkpoint saving
* Best model selection
* Accuracy evaluation

---

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

---

## Docker Deployment

Build image:

```bash
docker build -t sps-genai-cnn .
```

Run container:

```bash
docker run -p 8000:80 sps-genai-cnn
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Project Structure

```text
sps_genai
│
├── app
│   ├── __init__.py
│   ├── bigram_model.py
│   └── embedding_model.py
│
├── helper_lib
│   ├── data_loader.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── checkpoints.py
│   └── utils.py
│
├── checkpoints
│
├── train_cnn.py
├── main.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```
