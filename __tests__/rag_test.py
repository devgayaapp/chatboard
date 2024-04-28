from typing import List
from pydantic import BaseModel, Field
import pytest

from chatboard.text.vectors.embeddings.text_embeddings import DenseEmbeddings
from chatboard.text.vectors.rag_documents import RagDocuments
from chatboard.text.vectors.stores.qdrant_vector_store import QdrantVectorStore
from chatboard.text.vectors.vectorizers.base import VectorMetrics, VectorizerBase



@pytest.mark.asyncio
async def test_text_rag():
    
    test_texts = [
        "this is a test 1",
        "another test 2"
    ]

    rag_space = RagDocuments("text_rag")
    await rag_space.create_namespace()
    await rag_space.add_documents(test_texts, test_texts)

    results = await rag_space.similarity("another", top_k=1)
    assert len(results) == 1
    assert results[0].metadata.value == "another test 2"
    await rag_space.delete_namespace()



class Post(BaseModel):
    id: int
    title: str
    content: str

    def __str__(self):
        return f"{self.title}\n{self.content}"

class LabelOutput(BaseModel):
    Classification: str
    Justification: str
    Confidence: float


class PostVectorizer(VectorizerBase):
    name: str = "post-dense"
    size: int = 1536
    dense_embeddings: DenseEmbeddings = Field(default_factory=DenseEmbeddings)
    metric: VectorMetrics = VectorMetrics.COSINE
    
    async def embed_documents(self, documents: List[Post]):
        return await self.dense_embeddings.embed_documents([str(d) for d in documents])
    
    async def embed_query(self, query: Post):
        return await self.dense_embeddings.embed_query(str(query))


@pytest.mark.asyncio
async def test_post_rag():
    test_posts = [
        Post(id=1, title="test 1", content="this is a test 1"),
        Post(id=2, title="test 2", content="another test 2")
    ]
    test_values = [
        LabelOutput(Classification="test", Justification="test", Confidence=0.5),
        LabelOutput(Classification="test", Justification="test", Confidence=0.5)
    ]
    vectorizer = PostVectorizer()
    vector_store = QdrantVectorStore("test_post_rag")
    rag_space = RagDocuments(
        "test_post_rag", 
        vectorizers=[vectorizer], 
        vector_store=vector_store,
        key_class=Post,
        value_class=LabelOutput
    )

    await rag_space.create_namespace()
    await rag_space.add_documents(test_posts, test_values)

    results = await rag_space.similarity(Post(id=2, title="test 2", content="another"), top_k=1)
    assert len(results) == 1
    assert results[0].metadata.value.Classification == "test"
    await rag_space.delete_namespace()
