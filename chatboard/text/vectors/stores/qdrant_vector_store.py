from typing import Any, Dict, List, TypeVar, Union, Optional, Generic
from uuid import uuid4
from pydantic import BaseModel
from chatboard.text.vectors.stores.base import VectorStoreBase
from chatboard.text.vectors.vectorizers.base import VectorMetrics, VectorizerBase
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from qdrant_client.models import PointStruct, SparseVector, FieldCondition, Range, Filter
from qdrant_client.models import NamedSparseVector, DatetimeRange, SearchRequest, NamedVector, SparseVector
from qdrant_client import models
from ..utils import chunks
import os



class VectorSearchResult(BaseModel):
    id: str
    score: float
    metadata: Any
    vector: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True

class VectorStoreException(Exception):
    pass


def metrics_to_qdrant(metric: VectorMetrics):
    if metric == VectorMetrics.COSINE:
        return Distance.COSINE
    elif metric == VectorMetrics.EUCLIDEAN:
        return Distance.EUCLIDEAN
    elif metric == VectorMetrics.MANHATTAN:
        return Distance.MANHATTAN
    else:
        raise ValueError(f"Unsupported metric {metric}")

T = TypeVar('T', bound=BaseModel)
U = TypeVar('U', bound=VectorizerBase)

# class QdrantVectorStore(Generic[T, U]):
class QdrantVectorStore(VectorStoreBase):

    def __init__(
        self,
        collection_name: str,
        url: str | None=None,
        api_key: str | None = None,
        ) -> None:
        url = url or os.environ.get("QDRANT_URL")
        api_key = api_key or os.environ.get("QDRANT_API_KEY", None)
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
            # host=os.environ['QDRANT_HOST'], 
            # port=os.environ['QDRANT_PORT']
        )
        self.collection_name = collection_name


    def _pack_points(self, points):
        recs = []
        for p in points:
            rec = VectorSearchResult(
                id=p.id,
                score=p.score if hasattr(p, "score") else -1,
                metadata=p.payload,
                vector=p.vector
            )
            recs.append(rec)
        return recs

    

    # async def similarity(self, query, top_k=3, filters=None, alpha=None):        
    #     query_filter = None
    #     if filters is not None:
    #         must_not = filters.get('must_not', None)
    #         must = filters.get('must', None)
    #         query_filter = models.Filter(
    #             must_not=must_not,
    #             must=must
    #         )

    #     recs = await self.client.search(
    #         collection_name=self.collection_name,
    #         query_vector=NamedVector(
    #             name="posts-dense",
    #             vector=query
    #         ),
    #         query_filter=query_filter,
    #         limit=top_k,            
    #         with_payload=True,
    #     )
    #     return self._pack_points(recs)    

    async def similarity(self, query, top_k=3, filters=None, alpha=None, with_vectors=False):        
        query_filter = None
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            # must_not = filters.get('must_not', None)
            # must = filters.get('must', None)
            query_filter = models.Filter(
                must_not=must_not,
                must=must
            )
        if len(query.items()) == 1:
            vector_name, vector_value = list(query.items())[0]
            recs = await self.client.search(
                collection_name=self.collection_name,
                query_vector=NamedVector(
                    name=vector_name,
                    vector=vector_value
                ),
                query_filter=query_filter,
                limit=top_k,            
                with_payload=True,
                with_vectors=with_vectors,
            )
            return self._pack_points(recs)



    async def add_documents(self, vectors, metadata: List[Dict | BaseModel], ids=None, namespace=None, batch_size=100):
        namespace = namespace or self.collection_name
        # metadata = [m.dict(exclude={'__orig_class__'}) if isinstance(m, BaseModel) else m for m in metadata]
        metadata = [m if isinstance(m, dict) else m.dict(exclude={'__orig_class__'}) for m in metadata]
        if not ids:
            ids = [str(uuid4()) for i in range(len(vectors))]
        
        for vector_chunk in chunks(zip(ids, vectors, metadata), batch_size=batch_size):
            points = [
                PointStruct(
                    id=id_,
                    payload=meta,
                    vector=vec
                )
                for id_, vec, meta in vector_chunk]
            await self.client.upsert(
                collection_name=namespace,
                points=points
            )


    async def create_collection(self, vectorizers, collection_name: str | None = None):
        collection_name=collection_name or self.collection_name
        vector_config = {}
        for vectorizer in vectorizers:
            vector_config[vectorizer.name] = VectorParams(size=vectorizer.size, distance=metrics_to_qdrant(vectorizer.metric))
        await self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=vector_config            
        )


    async def delete_collection(self, collection_name: str | None =None):
        collection_name = collection_name or self.collection_name
        await self.client.delete_collection(collection_name=collection_name)



    # async def delete_documents_ids(self, ids: List[Union[str, int]]):
    async def delete_documents_ids(self, ids: List[str | int]):
        res = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids,
            )
        )
        return res

    async def delete_documents(self, filters):
        if not filters:
            raise ValueError("filters must be provided. to delete all documents, use delete_collection method.")
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            filter_ = models.Filter(
                must_not=must_not,
                must=must
            )
        
        res = await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=filter_
            )
        )
        return res


    async def get_documents(self, filters: Any,  ids: List[str | int] | None=None, top_k: int=10, with_payload=False, with_vectors=False):
        filter_ = None
        if ids is not None:
            # top_k: int | None = None
            filter_ = models.Filter(
                must=[
                    models.HasIdCondition(has_id=ids)
                ],
            )
        if filters is not None:
            must_not, must = self.parse_filter(filters)
            filter_ = models.Filter(
                must_not=must_not,
                must=must
            )
        recs, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return self._pack_points(recs)


    async def get_many(self, ids: List[str | int] | None=None, top_k=10, with_payload=False, with_vectors=False):
        filter_ = None
        if ids is not None:
            top_k = None
            filter_ = models.Filter(
                must=[
                    models.HasIdCondition(has_id=ids)
                ],
            )
        recs, _ = await self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_,
            limit=top_k,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return self._pack_points(recs)


    async def info(self):        
        return await self.client.get_collection(self.collection_name)
        
            


    def parse_filter(self, filters: Dict[str, Any]):

        must = []
        must_not = []

        def is_range(value):    
            for k, v in value.items():
                if k not in ["$gt", "$gte", "$lt", "$lte"]:
                    return False
            return True

        def unpack_range(value):
            models.Range(
                gt=value.get('$gt', None),
                gte=value.get('$gte', None),
                lt=value.get('$lt', None),
                lte=value.get('$lte', None),
            )

        for field, value in filters.items():
            if type(value) == dict:
                if is_range(value):
                    must.append(models.FieldCondition(key=field, range=unpack_range(value)))
                else:
                    for k, v in value.items():
                        if k == "$ne":
                            must_not.append(models.FieldCondition(key=field, match=models.MatchValue(value=v)))
                        elif k == "$eq":
                            must.append(models.FieldCondition(key=field, match=models.MatchValue(value=v)))
            else:
                must.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))

        return must_not, must

# class QdrantVectorStore():

#     def __init__(
#         self,
#         collection_name: str,
#         metadata_model: BaseModel=None,
#         vectorizers: List[VectorizerBase]=None,
#         ) -> None:
#         self.client = AsyncQdrantClient(
#             url=os.environ.get("QDRANT_URL"),
#             api_key=os.environ.get("QDRANT_API_KEY", None),
#             # host=os.environ['QDRANT_HOST'], 
#             # port=os.environ['QDRANT_PORT']
#         )
#         self.collection_name = collection_name
#         self.metadata_model = metadata_model
#         self.vectorizers = vectorizers


#     def _pack_points(self, points):
#         recs = []
#         for p in points:
#             rec = VectorSearchResult(
#                 id=p.id,
#                 score=p.score if hasattr(p, "score") else -1,
#                 payload=p.payload
#             )
#             recs.append(rec)
#         return recs

    

#     async def similarity(self, query, top_k=3, filters=None, alpha=None):        
#         query_filter = None
#         if filters is not None:
#             must_not = filters.get('must_not', None)
#             must = filters.get('must', None)
#             query_filter = models.Filter(
#                 must_not=must_not,
#                 must=must
#             )

#         recs = await self.client.search(
#             collection_name=self.collection_name,
#             query_vector=NamedVector(
#                 name="posts-dense",
#                 vector=query
#             ),
#             query_filter=query_filter,
#             limit=top_k,            
#             with_payload=True,
#         )
#         return self._pack_points(recs)




#     async def _add_documents(self, embeddings, metadata: List[Union[Dict, BaseModel]], ids=None, namespace=None, batch_size=100):
#         metadata = [m.dict() if isinstance(m, BaseModel) else m for m in metadata]
#         if not ids:
#             ids = [str(uuid4()) for i in range(len(embeddings))]
        
#         for vectors in chunks(zip(ids, embeddings, metadata), batch_size=batch_size):
#             points = [
#                 PointStruct(
#                     id=id_,
#                     payload=meta,
#                     vector={
#                         "dense_emb": emb
#                     }
#                 )
#                 for id_, emb, meta in vectors]
#             await self.client.upsert(
#                 collection_name=self.collection_name,
#                 points=points
#             )


#     async def create_collection(self, collection_name: str =None):
#         collection_name=collection_name or self.collection_name
#         vector_config = {}
#         for vectorizer in self.vectorizers:
#             vector_config[vectorizer.name] = VectorParams(size=vectorizer.size, distance=metrics_to_qdrant(vectorizer.metric))
#         await self.client.recreate_collection(
#             collection_name=collection_name,
#             vectors_config=vector_config            
#         )


#     async def delete_collection(self, collection_name: str=None):
#         collection_name = collection_name or self.collection_name
#         await self.client.delete_collection(collection_name=collection_name)



#     async def get_many(self, ids: List[str]=None, top_k=10, with_payload=False, with_vectors=False):
#         filter_ = None
#         if ids is not None:
#             top_k = None
#             filter_ = models.Filter(
#                 must=[
#                     models.HasIdCondition(has_id=ids)
#                 ],
#             )
#         recs, _ = await self.client.scroll(
#             collection_name=self.collection_name,
#             scroll_filter=filter_,
#             limit=top_k,
#             with_payload=with_payload,
#             with_vectors=with_vectors,
#         )
#         return self._pack_points(recs)