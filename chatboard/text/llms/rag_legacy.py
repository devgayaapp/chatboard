import asyncio
from typing import Union
from pydantic import BaseModel
from components.vectors.pinecone_text_vector_store import PineconeTextVectorStore
from components.vectors.embeddings.text_embeddings import DenseEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from uuid import uuid4



class RAGMetadata(BaseModel):
    key: str
    text: str
    feedback: Union[str, None] = None


class RAGExample:

    def __init__(self, id, key, metadata):
        self.id = id
        self.key = key
        self.text = metadata.text
        self.metadata = metadata

    def __repr__(self):
        return f"RAGExample\nid={self.id} \nkey={self.key}, \nmetadata={self.text})"
    
    def get_ai_message(self):
        return AIMessage(
            content=self.text,
            example=True
        )
    
    def get_user_message(self):
        return HumanMessage(
            content=self.key,
            example=True
        )

class MvpExample(RAGExample):
    
    def __init__(self, model, view, controller):
        ex_id = uuid4()
        super().__init__(ex_id, model, view)

    @property
    def model(self):
        return self.metadata
    
    @property
    def view(self):
        return self.text




class RagManager:

    def __init__(self, namespace, metadata_model=RAGMetadata, index='rag') -> None:
        embeddings = DenseEmbeddings()
        self.namespace = namespace
        self.vector_store = PineconeTextVectorStore(
            index, 
            namespace, 
            embeddings=embeddings, 
            metadata_model=metadata_model
        )
        self.metadata_model = metadata_model

    async def get_many(self, namespace: str=None):
        res = await asyncio.to_thread(
            self.vector_store.get_many,
            namespace=namespace or self.namespace
        )
        examples = []
        for ex in res:
            metadata = self.metadata_model(**ex['metadata'])
            examples.append(RAGExample(ex['id'], metadata.key, metadata))
        return examples

    async def search(self, query, top_k=3, namespace=None):
        namespace = namespace or self.namespace
        res = await self.vector_store.atext_similarity_search(
            query,
            top_k=top_k,
            namespace=namespace,
        )
        examples = []
        for ex in res:
            metadata = self.metadata_model(**ex['metadata'])
            examples.append(RAGExample(ex['id'], metadata.key, metadata))
        return examples

    
    async def add(self, key, metadata):
        output = await self.vector_store.aadd_documents(texts=[key], metadata=[metadata])
        return RAGExample(output['records'][0]['id'], metadata.key, metadata)

    async def add_many(self, examples: list[RAGMetadata]):
        metadata = []
        keys = []
        for m in examples:
            if isinstance(m, dict):
                m = self.metadata_model(**m)
            keys.append(m.key)
            if not isinstance(metadata, self.metadata_model):
                raise ValueError(f"metadata must be of type {self.metadata_model}")
            metadata.append(m.model_dump())
        
        output = await self.vector_store.aadd_documents(texts=keys, metadata=metadata)
        return output
        # [self.metadata_model(id, key, metadata) for id, metadata in res]
        # return [RAGExample(id, key, metadata) for id, metadata in res]

    async def delete_all(self):
        self.vector_store.delete(delete_all=True)


    async def delete(self,id):
        self.vector_store.delete(id)


