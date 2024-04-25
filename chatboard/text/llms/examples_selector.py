from typing import Dict, List
from uuid import uuid4
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.example_selector.base import BaseExampleSelector

import pinecone
from pyparsing import Any
from config import PINECONE_ENV, PINECONE_KEY, OPENAI_API_KEY



embedding_models = {
    'text-embedding-ada-002': {
        'model_name': 'text-embedding-ada-002',
        'dimension': 1536,
        'api_key': PINECONE_KEY
    }
}

pinecone.init(
    api_key= PINECONE_KEY,
    environment=PINECONE_ENV
)


class PineconeExampleSelector(BaseExampleSelector):

    def __init__(self, index_name: str, embed_model_name: str = 'text-embedding-ada-002') -> None:
        
        self.embed = OpenAIEmbeddings(
            document_model_name=embedding_models[embed_model_name]['model_name'],
            query_model_name=embedding_models[embed_model_name]['model_name'],
            openai_api_key=embedding_models[embed_model_name]['api_key']
        )        
        self.index = pinecone.Index(index_name)
        self.vectorstore = Pinecone(self.index, self.embed.embed_query, 'key')


    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        input_variables['key']
        """Select which examples to use based on the inputs."""

    def add_example(self, example: Dict[str, str]) -> Any:
        return self.add_vectors([example['key']], [example['text']])


    async def similarity(self, query: str, k: int=2) -> None:
        return await self.vectorstore.similarity_search(query, k=k)
       
    async def add_vectors(self, texts: List[str], metadatas: List[Any]) -> None:
        return await self.vectorstore.aadd_texts(texts=texts, metadatas=metadatas)
        # ids = [str(uuid4()) for _ in range(len(examples))]
        # example_embeddings = self.embed.embed_documents(keys)
        
        # vectors = zip(ids, example_embeddings, examples)
        # self.index.upsert(vectors=vectors)
        # return vectors


    def get_vectors() -> None:
        pass


    @staticmethod
    def create_index(index_name: str, metric='dotproduct', embed_model_name = 'text-embedding-ada-002') -> None:
        return pinecone.create_index(
            name=index_name,
            metric=metric,
            dimension=embedding_models[embed_model_name]['dimension']
        )

    @staticmethod
    def delete_index(index_name: str) -> None:
        pinecone.delete_index(index_name)


    @staticmethod
    def list_indexes() -> None:
        return pinecone.list_indexes()
