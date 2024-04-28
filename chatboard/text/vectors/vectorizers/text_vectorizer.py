from typing import Any, List
from chatboard.text.vectors.embeddings.text_embeddings import DenseEmbeddings
from chatboard.text.vectors.vectorizers.base import VectorMetrics, VectorizerBase
from pydantic import BaseModel, Field




class TextVectorizer(VectorizerBase):
    name: str = "dense"
    size: int = 1536
    dense_embeddings: DenseEmbeddings = Field(default_factory=DenseEmbeddings)
    metric: VectorMetrics = VectorMetrics.COSINE
    
    async def embed_documents(self, documents: List[str]):
        return await self.dense_embeddings.embed_documents(documents)
    
    async def embed_query(self, query: str):
        return await self.dense_embeddings.embed_query(query)