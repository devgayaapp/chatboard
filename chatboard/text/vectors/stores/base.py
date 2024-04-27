from abc import abstractmethod
from typing import Any, List







class VectorStoreBase:



    @abstractmethod
    async def add_documents(self, embeddings, metadata: List[Any], ids=None, namespace=None, batch_size=100):
        """Adds documents to the vector store"""
        raise NotImplementedError
    
    @abstractmethod
    async def similarity(self, query, top_k=3, filters=None, alpha=None):
        """Searches for similar documents in the vector store"""
        raise NotImplementedError


    @abstractmethod
    async def create_collection(self):
        """Creates a collection in the vector store"""
        raise NotImplementedError


    async def delete_collection(self):
        """Deletes a collection in the vector store"""
        raise NotImplementedError