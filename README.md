# Embedding_Inference

![CI Status](https://github.com/AmrAhmedElagoz/Embedding_Inference/actions/workflows/test.yml/badge.svg)

A lightweight Python package for performing embedding inference efficiently using FastAPI and Stella embeddings.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Install from PyPI via Git](#install-from-pypi-via-git)
  - [Install from Source](#install-from-source)
  - [Editable Installation](#editable-installation)
  - [Development Installation](#development-installation)
- [Usage](#usage)
  - [Running the API](#running-the-api)
  - [API Endpoints](#api-endpoints)
  - [Python Client Example](#python-client-example)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **FastAPI-based REST API** for embedding inference
- **Stella 400M v5 model** for high-quality embeddings
- **Single and batch processing** support
- **CUDA acceleration** for fast inference
- **Normalized embeddings** for similarity calculations

---

## Installation

### Install from PyPI via Git

```bash
pip install git+https://github.com/AmrAhmedElagoz/Embedding_Inference.git
```

### Install from Source

Clone the repository and install the package:

```bash
git clone https://github.com/AmrAhmedElagoz/Embedding_Inference.git
cd Embedding_Inference
pip install .
```

### Editable Installation

For development and direct editing:

```bash
git clone https://github.com/AmrAhmedElagoz/Embedding_Inference.git
cd Embedding_Inference
pip install -e .
```

### Development Installation

Includes development dependencies (for testing, linting, etc.):

```bash
git clone https://github.com/AmrAhmedElagoz/Embedding_Inference.git
cd Embedding_Inference
pip install -e .[dev]
```

---

## Usage

### Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### API Endpoints

#### Health Check

**GET** `/`

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Embedding API is running"
}
```

#### Single Text Embedding

**POST** `/embed`

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Your input text here"}'
```

Response:
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

#### Batch Text Embedding

**POST** `/embedBatch`

```bash
curl -X POST http://localhost:8000/embedBatch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["First text", "Second text", "Third text"]}'
```

Response:
```json
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.234, -0.567, ...],
    [0.345, -0.678, ...]
  ]
}
```

### Python Client Example

```python
import requests

# Single embedding
response = requests.post(
    "http://localhost:8000/embed",
    json={"text": "This is a sample text"}
)
embedding = response.json()["embedding"]
print(f"Embedding dimension: {len(embedding)}")

# Batch embeddings
response = requests.post(
    "http://localhost:8000/embedBatch",
    json={"texts": ["First text", "Second text", "Third text"]}
)
embeddings = response.json()["embeddings"]
print(f"Number of embeddings: {len(embeddings)}")
```

---

## Configuration

The embedding model is configured with the following parameters:

- **Model**: `dunzhang/stella_en_400M_v5`
- **Device**: CUDA (GPU acceleration)
- **Normalization**: Enabled for embeddings
- **Progress Display**: Enabled

To modify the configuration, edit the `EmbbederArgs` in `main.py`:

```python
args = EmbbederArgs(
    model_name='dunzhang/stella_en_400M_v5',
    show_progress=True,
    model_kwargs={'device': 'cuda', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)
```