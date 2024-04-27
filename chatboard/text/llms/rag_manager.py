from abc import abstractmethod
import asyncio
from typing import Generic, Any, Optional, TypeVar, Union
from pydantic import BaseModel
from pydantic.generics import GenericModel
import numpy as np
# from ...common.vectors.embeddings.text_embeddings import DenseEmbeddings, Embeddings
from ..vectors.embeddings.text_embeddings import DenseEmbeddings, Embeddings
from uuid import uuid4
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import numpy as np


# from ...common.vectors.pinecone_vector_store import PineconeVectorStore
from ..vectors.stores.pinecone_vector_store import PineconeVectorStore


def values_to_csr_matrix(sparse_values, shape):
    row_indices = [0 for _ in sparse_values['indices']]
    col_indices = sparse_values['indices']
    data = sparse_values['values']
    return csr_matrix((data, (row_indices, col_indices)), shape=(1, shape))



def get_diverse_vectors(examples, shape):
    sparse_vecs = [values_to_csr_matrix(ex.embedding.sparse, shape) for ex in examples]

    if len(sparse_vecs) == 0:
        return examples
    
    mean_sparse = sum(sparse_vecs) / len(sparse_vecs)

    diff_vecs = []
    for vec in sparse_vecs:
        diff_vecs.append(norm(vec - mean_sparse))
    scored_examples = list(zip(examples, diff_vecs))
    scored_examples.sort(key=lambda x: x[1], reverse=True)
    return scored_examples



def filter_std_examples(examples, std=2):
    ex_std = np.std([e.score for e in examples])
    max_value = np.max([e.score for e in examples])
    mean_values = np.mean([e.score for e in examples])
    best_examples = [e for e in examples if e.score > mean_values - ex_std * std]
    return best_examples


def filter_iqr_examples(examples):
    scores = [e.score for e in examples]
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [e for e in examples if e.score > lower_bound]

# class ChatRagExample(BaseModel):
#     id: Optional[str] = None
#     inputs: str
#     output: str
#     feedback: Union[str, None] = None


class NumpyArray(np.ndarray):
    """
    A custom type that represents a NumPy array.
    This is needed to help Pydantic recognize and validate NumPy arrays.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError('numpy.ndarray required')
        return v


# class RagKey(BaseModel):
    # embeddings: NumpyArray
    # sparse_embeddings: Optional[NumpyArray] = None



class RagValue(BaseModel):
    # feedback: Union[str, None] = None
    key: str
    value: str

# class RagVector(BaseModel):
#     key: Embeddings
#     value: RagValue
    

ValueT = TypeVar('ValueT', bound=RagValue)

class RagVector(GenericModel, Generic[ValueT]):
    id: Optional[Union[str, int]] = None
    score: Optional[float] = None
    embedding: Optional[Embeddings] = None
    metadata: ValueT

    def to_dict(self):
        if hasattr(self.metadata, 'to_dict'):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata.dict()
        return {
            "id": self.id,
            "score": self.score,
            "metadata": metadata
        }



# class RAGExample:

#     def __init__(self, id, metadata, feedback=None):
#         self.id = id
#         self.metadata = metadata
#         self.feedback = feedback

#     def __repr__(self):
#         return f"RAGExample\nid={self.id} \nmetadata={self.metadata})"
    
#     def get_ai_message(self):
#         return AIMessage(
#             content=self.text,
#             example=True
#         )
    
#     def get_user_message(self):
#         return HumanMessage(
#             content=self.key,
#             example=True
#         )


# class MvpExample(RAGExample):
    
#     def __init__(self, model, view, controller):
#         ex_id = uuid4()
#         super().__init__(ex_id, model, view)

#     @property
#     def model(self):
#         return self.metadata
    
#     @property
#     def view(self):
#         return self.text


class VectorizerBase:

    @abstractmethod
    def embed_documents(self, object):
        pass

    @abstractmethod
    def embed_query(self, object):
        pass





class RagVectorSpace:
    """
    this class is concerned only with key, value. it does not care about the how the key is vectorized.
    """


    def __init__(self, namespace, vectorizer: VectorizerBase, value_model=None, vector_store=None, index='rag') -> None:
        self.namespace = namespace
        self.vectorizer = vectorizer
        if vector_store is None:
            self.vector_store = PineconeVectorStore(
                index, 
                namespace, 
                metadata_model=value_model
            )
        else:
            self.vector_store = vector_store
        self.value_model = value_model


    def valudate_value(self, value):
        if type(value) == dict:
            return RagVector[self.value_model](**value)
        elif type(value.metadata) != self.value_model:
            raise ValueError(f"metadata must be of type {self.value_model}")
        return value
    
    def valudate_query(self, value):
        if type(value) == dict:
            return self.value_model(**value)
        elif type(value) != self.value_model:
            raise ValueError(f"metadata must be of type {self.value_model}")
        return value


    async def get_many(self, top_k=100, namespace: str=None, return_key=False):
        results = await self.vector_store.get_many(
            top_k=top_k,
            namespace=namespace or self.namespace,
            # parse_model=False,
            include_values=return_key
        )
        examples = []
        for res in results:
            # metadata, feedback = self._unpack_value(res)
            # examples.append(RagVector[self.value_model](
            #     id=res.id, 
            #     score=res.score,
            #     key=res.embedding, 
            #     value=metadata, 
            #     feedback=feedback
            # ))
            # examples.append(RagVector[self.value_model](
            #     id=res.id,
            #     metadata=self.valudate_value(**res.metadata)
            # ))
            examples.append(RagVector[self.value_model](
                id=res.id,
                metadata=res.metadata
            ))
        return examples


    async def similarity(self, query, top_k=3, alpha=None, namespace=None, use_diverse=False, return_key=False):
        # query = self.valudate_query(query)
        query_embs = await self.vectorizer.embed_query(query)
        namespace = namespace or self.namespace
        acutual_top_k = top_k
        if use_diverse:
            acutual_top_k = max(10, 2*top_k)
            return_key = True
        results = await self.vector_store.similarity(
            query_embs,
            top_k=acutual_top_k,
            alpha=alpha,
            namespace=namespace,
            # parse_model=False,
            include_values=return_key
        )
        examples = []
        for res in results:
            vec = RagVector[self.value_model](
                id=res.id,                
                score=res.score, 
                metadata=res.metadata, 
            )
            if return_key:
                vec.embedding = res.embedding
            
            examples.append(vec)
            # examples.append(RagVector[self.value_model](                
            #     id=res.id,                
            #     score=res.score, 
            #     key=res.embedding,
            #     value=metadata, 
            #     feedback=feedback
            # ))

        if use_diverse:
            best_examples = filter_iqr_examples(examples)
            if len(best_examples) < top_k:
                return examples[:top_k]
            diverse_examples = get_diverse_vectors(
                best_examples, 
                self.vectorizer.sparse_embeddings.get_shape()
            )
            return  [e[0] for e in diverse_examples][:top_k]

        return examples


    def _unpack_value(self, ex):
        feedback = None
        metadata = ex.metadata
        if "feedback" in metadata:
            feedback = metadata.pop('feedback')
        if self.value_model is not None:
            metadata = self.value_model(**metadata)
        return metadata, feedback
    

    # async def add(self, example):
    #     embeddings = self.vectorizer.vectorize(example)
    #     output = await self.vector_store.aadd_documents(texts=[key], metadata=[metadata])
    #     return RAGExample(output['records'][0]['id'], key, metadata)

    async def update(self, example_id, metadata):
        if type(metadata) == dict:
            metadata = self.value_model(**metadata)
        res = await self.vector_store.update(example_id, metadata)
        return res


    async def add_many(self, examples: list[Any]):
        examples = [self.valudate_value(e) for e in examples]
        metadata_list = [e.metadata for e in examples]
        key_list = await self.vectorizer.embed_documents(metadata_list)
        id_list = [e.id if hasattr(e, 'id') else str(uuid4()) for e in examples]
        output = await self.vector_store.add_documents(embeddings=key_list, metadata=metadata_list, ids=id_list)
        return output
    

    async def delete_all(self):
        self.vector_store.delete(delete_all=True)


    async def delete(self, example_id):
        return await self.vector_store.adelete(example_id)
    

    async def get_namespaces(self):
        return self.vector_store.index.describe_index_stats()['namespaces']


