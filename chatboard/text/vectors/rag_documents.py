from typing import Any, Dict, List, TypeVar, Union, Optional, Generic, Type
from uuid import uuid4
from pydantic import BaseModel
from chatboard.text.vectors.stores.base import VectorStoreBase
from chatboard.text.vectors.stores.qdrant_vector_store import QdrantVectorStore
from chatboard.text.vectors.vectorizers.base import VectorMetrics, VectorizerBase
import asyncio

from chatboard.text.vectors.vectorizers.text_vectorizer import TextVectorizer



K = TypeVar('K', bound=BaseModel)
V = TypeVar('V', bound=BaseModel)



class RagDocMetadata(Generic[K, V], BaseModel):
    id: Union[str, int]
    key: Union[K, str]
    value: Union[V, str]



# class Point(BaseModel):
#     id: str
#     vector: Dict[str, Any]
#     metadata: RagDocMetadata[K, V]

class RagSearchResult(BaseModel):
    id: str
    score: float
    metadata: RagDocMetadata[K, V]


class RagDocuments(Generic[K, V]):

    def __init__(
            self, 
            namespace: str, 
            vectorizers: List[VectorizerBase] = [], 
            vector_store: VectorStoreBase = None, 
            key_class: Union[Type[K], str] = str, 
            value_class: Union[Type[V], str] = str
        ) -> None:
        self.namespace = namespace
        if vector_store is None:
            self.vector_store = QdrantVectorStore(namespace)
        else:
            self.vector_store = vector_store
        if not vectorizers:
            self.vectorizers = [TextVectorizer()]
        else:
            self.vectorizers = vectorizers
        self.key_class = key_class
        self.value_class = value_class

    async def _embed_documents(self, documents: List[K]):
        embeds = await asyncio.gather(
            *[vectorizer.embed_documents(documents) for vectorizer in self.vectorizers]
        )
        vectors = []
        embeds_lookup = {vectorizer.name: embed for vectorizer, embed in zip(self.vectorizers, embeds)}
        for i in range(len(documents)):
            vec = {}
            for vectorizer in self.vectorizers:
                vec[vectorizer.name] = embeds_lookup[vectorizer.name][i]
            vectors.append(vec)
        return vectors
    
    def _pack_results(self, results):
        rag_results = []
        for res in results:
            if self.key_class == str:
                key = res.metadata["key"]
            else:
                key = self.key_class(**res.metadata["key"])
            if self.value_class == str:
                value = res.metadata["value"]
            else:
                value = self.value_class(**res.metadata["value"])
            rag_results.append(RagSearchResult(
                id=res.id, 
                score=res.score, 
                metadata=RagDocMetadata[self.key_class, self.value_class](
                    id=res.id, 
                    key=key, 
                    value=value
                )
            ))
        return rag_results
        # return [RagDocMetadata(id=res.id, key=res.key, value=res.value) for res in results]


    async def add_documents(self, keys: List[K], values: List[V], ids=None):
        vectors = await self._embed_documents(keys)
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(keys))]
        documents = [RagDocMetadata[self.key_class, self.value_class](id=i, key=key, value=value) for i, key, value in zip(ids, keys, values)]
        outputs = await self.vector_store.add_documents(vectors, documents, namespace=self.namespace)
        return outputs


    async def similarity(self, query: K, top_k=3, filters=None, alpha=None):
        query_vector = await self._embed_documents([query])
        query_vector = query_vector[0]
        res = await self.vector_store.similarity(query_vector, top_k, filters, alpha)
        return self._pack_results(res)


    async def create_namespace(self, namespace: str=None):
        namespace = namespace or self.namespace
        return await self.vector_store.create_collection(self.vectorizers, namespace)
    
    async def delete_namespace(self, namespace: str=None):
        namespace = namespace or self.namespace
        return await self.vector_store.delete_collection(namespace)