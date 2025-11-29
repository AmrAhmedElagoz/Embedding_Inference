from fastapi import FastAPI, Body
from pydantic import BaseModel
from localEmb.config import EmbbederArgs
from localEmb.embedder import Embedder
from typing import List

args = EmbbederArgs(
    model_name='dunzhang/stella_en_400M_v5',
    show_progress=True,
    model_kwargs={'device': 'cuda', 'trust_remote_code': True},
    encode_kwargs={'normalize_embeddings': True}
)

embedder = Embedder(args)

app = FastAPI()

class TextInput(BaseModel):
    text: str

class BatchTextInput(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {"message": "Embedding API is running"}

@app.post("/embed")
def embed_text(data: TextInput):
    embedding = embedder.create_embedding(data.text)
    return {"embedding": embedding}

@app.post("/embedBatch")
def embed_batch(data: BatchTextInput):
    embeddings = embedder.create_embedding(data.texts)
    return {"embeddings": embeddings}