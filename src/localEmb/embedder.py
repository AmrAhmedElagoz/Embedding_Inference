from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from .config import EmbbederArgs
from typing import Union, List


class Embedder:
    def __init__(self, args: EmbbederArgs):
        self.emb = HuggingFaceEmbeddings(
            model_name=args.model_name,
            model_kwargs=args.model_kwargs,
            encode_kwargs=args.encode_kwargs,
            show_progress=args.show_progress,
        )

    def create_embedding(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        return self.emb.embed_documents(texts)
