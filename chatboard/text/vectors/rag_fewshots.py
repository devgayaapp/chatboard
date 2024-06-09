
from typing import Any, List
from chatboard.text.llms.conversation import HumanMessage, AIMessage
from chatboard.text.vectors.rag_documents2 import RagDocuments
from pydantic import BaseModel




class RagFewshots():

    def __init__(self, namespace, key_class=str, value_class=str) -> None:
        self.key_class = key_class
        self.value_class = value_class

        class ExampleShot(BaseModel):
            key: key_class
            value: value_class

        self.metadata_class = ExampleShot
        self.rag_documents = RagDocuments(namespace, metadata_class=ExampleShot)


    async def get_examples(self, key, top_k=5):
        results = await self.rag_documents.similarity(key, top_k=5)
        return self._pack_messages(results)
    
    async def add_examples(self, key, value, id=None):
        if id:
            return await self.rag_documents.add_documents([key], [value], [id])
        return await self.rag_documents.add_documents([key], [value])
    
    def _pack_messages(self, results):
        examples = []
        for res in results:
            examples.append(HumanMessage(content=res.metadata.key, example=True))
            examples.append(AIMessage(content=res.metadata.value, example=True))
        return examples
    
    async def get_many(self, top_k=10):
        results = await self.rag_documents.get_many(top_k=top_k)        
        return self._pack_messages(results)
