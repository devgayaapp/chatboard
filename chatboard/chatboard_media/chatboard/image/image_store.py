import asyncio
from typing import Any, Dict, List, Union
from datetime import datetime
import numpy as np
from pydantic import BaseModel
from components.image.detection_client import Face
from components.image.image import Image
from components.media.media import Media
from components.storage.bucket import Bucket
from components.vectors.pinecone_image_vector_store import PineconeImageVectorStore
from components.backend import Backend, MediaBERec, MediaType
from components.video.video import Video
from config import AWS_IMAGE_BUCKET, AWS_THUMBNAIL_BUCKET, PINECONE_IMAGE_INDEX, PINECONE_NAMESPACE
from util.boto import delete_s3_obj


class ImageStoreException(Exception):
    pass




# metadata model for vector store
class VectorMediaMetadata(BaseModel):
    description: str
    photo_type: str
    number_of_images: int
    source: str
    media_type: str
    event_idx: Union[int, None] = -1
    video_start_at: Union[float, None] = 0
    duration: Union[float, None] = 0



class MediaSearchResult:

    def __init__(self, media, score):
        self.score = score
        self.media = media

    # def __str__(self) -> str:
    #     return f"{self.score} - {self.media.filename}"
    
    # def __repr__(self) -> str:
    #     return f"{self.score} - {self.media.filename}"
    def __repr__(self) -> str:
        return f"{self.score} - {self.media.metadata.description}"

    def _repr_png_(self):
        print("score:", self.score)
        return self.media.thumbnail._repr_png_()
    
    def to_json(self):
        return {
            'score': self.score,
            'media': self.media.to_json()
        }
    
class MediaSearchResultList(list):

    def show(self):
        return Media.grid([m.media for m in self])
    
    def to_json(self):
        return [m.to_json() for m in self]

class ImageStore:


    def __init__(
            self, 
            index_name=PINECONE_IMAGE_INDEX, 
            namespace=PINECONE_NAMESPACE, 
            backend=None, 
            init_ai=True, 
            image_bucket=AWS_IMAGE_BUCKET, 
            thumbnail_bucket=AWS_THUMBNAIL_BUCKET,
            image_bucket_obj=None,
            thumbnail_bucket_obj=None,
            ) -> None:
        self._init_ai = init_ai
        #? initializing client for embeddings
        self.vector_store = PineconeImageVectorStore(index_name, namespace, init_ai=init_ai, metadata_model=VectorMediaMetadata)
        self.backend = backend if backend is not None else Backend()
        # self.image_bucket = image_bucket
        # self.thumbnail_bucket = thumbnail_bucket
        if image_bucket_obj:
            self.image_bucket = image_bucket_obj
        else:
            self.image_bucket = Bucket(image_bucket)
        if thumbnail_bucket_obj:
            self.thumbnail_bucket = thumbnail_bucket_obj
        else:
            self.thumbnail_bucket = Bucket(thumbnail_bucket)
    

    def add_image(
            self, 
            image: Image, 
            metadata: Any, 
            filename: str, 
            thumbnail_image: Image,
            thumbnail_filename: str= None,  
            verify_duplicate=True, 
            image_embeddings=None, 
            faces=None, 
            source="hasbara",
            media_id=None,
            url=None,        
        ):  
        backend_media_res = None
        vector_media_res = None
        if media_id is None:
            raise ImageStoreException("media_id cannot be None")
        try:
            #upload image to s3
            # image.to_s3(self.image_bucket, filename)
            self.image_bucket.add(image, filename)

            #generate thumbnail is it doesn't exist
            if thumbnail_image is None:
                thumbnail_image = image.get_thumbnail()
            if thumbnail_filename is None:
                thumbnail_filename = "thumbnail_" + "".join(filename.split('.')[:-1]) + '.png'
            # thumbnail_image.to_s3(self.thumbnail_bucket, thumbnail_filename)
            self.thumbnail_bucket.add(thumbnail_image, thumbnail_filename)

            # add media to backend
            backend_media_res = self.backend.add_media(
                media_id=media_id,
                name=filename, 
                type='IMAGE',
                source=source,
                width =image.width,
                height = image.height,
                thumbnail=thumbnail_filename,
                is_processed=True,
                movie_context="-1",
                account_id="-1",
                faces=faces,
                url=url
            )

            #add media to vector store and connect via backend id
            vector_media_res = self.vector_store.add_image(
                image, 
                metadata, 
                backend_id=backend_media_res['id'], 
                verify_duplicate=verify_duplicate, 
                image_embeddings=image_embeddings,
                image_id=media_id
            )

            return {
                'backend_res': backend_media_res,
                'vector_res': vector_media_res,
            }
        except Exception as e:
            if backend_media_res:
                self.backend.delete_media(backend_media_res['id'])
            if vector_media_res:
                self.vector_store.delete(vector_media_res['id'])
            raise e
    
    

    def add_video(
            self, 
            video: Video, 
            filename: str, 
            thumbnail_image: Image,
            thumbnail_filename: str =None, 
            verify_duplicate=True,
            source="hasbara", 
            media_id=None,
            url=None,
        ):
        backend_media_res = None
        vector_res_list = []
        try:
            if media_id is None:
                raise ImageStoreException("media_id cannot be None")
            
            # video.to_s3(self.image_bucket, filename)
            self.image_bucket.add(video, filename)

            #handle thumbnails
            if thumbnail_filename is None:
                thumbnail_filename = ''.join(filename.split('.')[:-1]) + '.jpg'
            # thumbnail_image.to_s3(self.thumbnail_bucket, thumbnail_filename)
            self.thumbnail_bucket.add(thumbnail_image, thumbnail_filename)


            backend_media_res = self.backend.add_media(
                media_id=media_id,
                name=filename, 
                type='VIDEO',
                source=source,
                width =video.width,
                height = video.height,
                thumbnail=thumbnail_filename,
                video_duration=video.duration,
                is_processed=True,
                # events=video.shots.to_json(),
                events=[e.to_json() for e in video.shots],
                movie_context="-1",
                account_id="-1",
                url=url
            )

            
            # store each shot in vector store as a separate image and connect via backend id
            for idx, shot in enumerate(video.shots):
                vector_res = self.vector_store.add_image(                    
                    shot.get_image(), 
                    # shot.get_metadata(), 
                    shot.metadata, 
                    backend_id=backend_media_res['id'], 
                    verify_duplicate=verify_duplicate, 
                    image_embeddings=shot.embeddings,
                    image_id=media_id+"_"+str(idx),
                )
                shot.vector_uuid = vector_res['id']
                vector_res_list.append(vector_res)

            return {
                'backend_res': backend_media_res,
                'vector_res_list': vector_res_list,
            }
        except Exception as e:
            # delete media from databases if something goes wrong
            if backend_media_res:
                self.backend.delete_media(backend_media_res['id'])
            if vector_res_list:
                for vec_res in vector_res_list:
                    self.vector_store.delete(vec_res['id'])
            raise e
    

    def add_media(self, media: Media, source="hasbara") -> Media:

        if media.type == 'VIDEO':
            video_add_res = self.add_video(
                video=media._media,
                filename=media.filename,
                thumbnail_image=media.thumbnail,
                thumbnail_filename=media.thumbnail_filename,
                media_id=media.gen_unique_id(),
                source=source,
                url=media.url,
            )
            print(video_add_res)
            media.backend_uuid = video_add_res['backend_res']['id']
            
        elif media.type == 'IMAGE':
            image_add_res = self.add_image(
                image=media._media, 
                metadata=media.metadata,
                filename=media.filename,
                thumbnail_image=media.thumbnail,
                thumbnail_filename=media.thumbnail_filename,
                image_embeddings=media.embeddings,
                faces=media.faces,
                media_id=media.gen_unique_id(),
                source=source,
                url=media.url,
            )
            media.backend_uuid = image_add_res['backend_res']['id']    
            media._pinecone_uuid = image_add_res['vector_res']['id']
        else:
            print("UNKNOWN MEDIA TYPE")
        return media

    

    def get_image_embeddings(self, image: Image):
        return self.vector_store.get_image_embeddings(
            image
        )
    
    
    def aget_image_embeddings(self, image: Image):
        return self.vector_store.aget_image_embeddings(
            image
        )


    async def aget_similar_image(self, image_embedding, duplicate_threshold=0.99, namespace=None):
        return await self.vector_store.aget_similar_image(image_embedding=image_embedding, duplicate_threshold=duplicate_threshold, namespace=namespace)


    def update_image(self, image_id: str, metadata: Any):
        vec_update = self.vector_store.update(image_id, metadata)
        vec_update = vec_update
        return vec_update


    # def get_many(self, top_k=100, fetch_backend=True, fetch_images=False, filter=None, fetch_lazy=True):
    #     vector_media = self.vector_store.get_many(top_k=top_k, filter=filter)
    #     vector_media = [r.to_dict() for r in vector_media]        
    #     for vm in vector_media:            
    #         if fetch_backend:
    #             vm = self._pupulate_backend_data(vm)
    #         if fetch_images:
    #             vm = self._populate_s3_data(vm, fetch_lazy)
    #     media_list = []
    #     for vm in vector_media:
    #         media = Media(
    #             media=vm.get('media_item', None),
    #             thumbnail=vm.get('thumbnail', None),
    #             # caption=
    #             platform_id=vm['metadata'].get('platform_id', None),
    #             source=vm['metadata'].get('source', None),
    #             date=vm['metadata'].get('date', None),
    #             size=vm['metadata'].get('size', None),
    #             filename=vm.get('name', None),
    #             thumbnail_filename=vm.get('thumbnail_filename', None),
    #             media_type="VIDEO" if vm['metadata'].get('type', None) == "video" else "IMAGE",
    #             # platform_type=
    #             # platform_group=
    #         )
    #         if media.type == 'IMAGE':
    #             media._pinecone_uuid = vm.get('id', None)
    #         else:
    #             print('not implemented')
    #             # media._pinecone_uuid = vm['metadata']['backend_id']
    #         media.backend_uuid = vm['metadata'].get('backend_id', None)
    #         media_list.append(media)
    #     return media_list
    async def get_many(self, top_k=100, fetch_backend=True, fetch_media=False, filter=None, fetch_lazy=True):
        # similar_images = self.vector_store.get_many(top_k=top_k, filter=filter)
        vector = np.array([0 for i in range(self.vector_store.dimension)]).reshape(1, -1)
        similar_images = self.vector_store.embedding_similarity_search(vector, filter=filter, top_k=top_k)
        media_list = MediaSearchResultList()
        for vec_media_rec in similar_images:
            media_item= None
            be_media_rec = None
            vec_metadata:VectorMediaMetadata = vec_media_rec.metadata
            media_rec = Media(
                media=None,
                media_type="VIDEO" if vec_metadata.media_type == "video" else "IMAGE",
                source= vec_metadata.source,
                caption=vec_metadata.description,
                backend_uuid=vec_media_rec.backend_id,
                metadata=vec_metadata,
            )
            media_rec.vector_rec = vec_media_rec
            if fetch_backend: 
                be_media_rec = await self.backend.aget_media(vec_media_rec.backend_id)
                media_rec.date = be_media_rec.create_at
                media_rec.filename = be_media_rec.name
                media_rec.thumbnail_filename = be_media_rec.thumbnail
                media_rec.faces = be_media_rec.faces
                media_rec.shots = be_media_rec.events
                media_rec.be_rec = be_media_rec
                
            if fetch_media:
                s3_data = await asyncio.to_thread(
                    self._fetch_s3_data, 
                    be_media_rec, 
                    fetch_lazy=True
                )
                media_rec._media = s3_data.get('media_item', None)
                media_rec.thumbnail = s3_data.get('thumbnail', None)
            media_rec = MediaSearchResult(media_rec, vec_media_rec.score)
            media_list.append(media_rec)
        return media_list

    
    def _pupulate_backend_data(self, media_dict):
        """
        appends backend data to data extracted from vector store
        Attributes:
            media_dict (dict): data extracted from vector store
            fetch_images (bool): whether to fetch image and thumbnail from s3
        """
        be_image_rec = self.backend.get_media(media_dict['metadata']['backend_id'])
        if be_image_rec is None:
            raise ImageStoreException(f"backend image record not found for {media_dict['metadata']['backend_id']}")
        media_dict.update(be_image_rec)    
        media_dict['thumbnail_filename'] = be_image_rec['thumbnail']
        del media_dict['thumbnail']
        
        return media_dict
    


    def _populate_s3_data(self, media_dict, fetch_lazy=True):
        if media_dict['type'] == 'IMAGE':
            if not fetch_lazy:
                media_dict['media_item'] = Image.from_bytes(self.image_bucket.get(media_dict['filename']))
            # media_dict['thumbnail'] = Image.from_s3(self.thumbnail_bucket, media_dict['thumbnail_filename'])
            media_dict['thumbnail'] = Image.from_bytes(self.thumbnail_bucket.get(media_dict['thumbnail_filename']))
        elif media_dict['type'] == 'VIDEO':
            if not fetch_lazy:
                media_dict['media_item'] = Image.from_bytes(self.image_bucket.get(media_dict['filename']))
            media_dict['thumbnail'] = Image.from_bytes(self.thumbnail_bucket.get(media_dict['thumbnail_filename']))
            # media_dict['thumbnail'] = Image.from_s3(self.thumbnail_bucket, media_dict['thumbnail_filename'])            
        return media_dict
    

    def _fetch_s3_data(self, be_media: MediaBERec, fetch_lazy=True):
        media_dict = {}
        if be_media.type == MediaType.IMAGE:
            if not fetch_lazy:
                media_dict['media_item'] = Image.from_bytes(self.image_bucket.get(be_media.name))
            # media_dict['thumbnail'] = Image.from_s3(self.thumbnail_bucket, media_dict['thumbnail_filename'])
            media_dict['thumbnail'] = Image.from_bytes(self.thumbnail_bucket.get(be_media.thumbnail))
        elif be_media.type == MediaType.VIDEO:
            if not fetch_lazy:
                media_dict['media_item'] = Image.from_bytes(self.image_bucket.get(be_media.name))
            media_dict['thumbnail'] = Image.from_bytes(self.thumbnail_bucket.get(be_media.thumbnail))
        else:
            raise ImageStoreException(f"unknown media type: {be_media.type}")
            # media_dict['thumbnail'] = Image.from_s3(self.thumbnail_bucket, media_dict['thumbnail_filename'])            
        return media_dict
    
    
    def get_images_by_text(self, text: str, filter= None, top_k=5, fetch_backend=True, fetch_images=False):
        similar_images = self.vector_store.text_similarity_search(text, filter=filter, top_k=top_k)
        images = []                    
        for si in similar_images:
            image = si.to_dict()
            image = self._pupulate_backend_data(image)
            if fetch_images:
                image = self._populate_s3_data(image)
            images.append(image)
        return images
    

    async def get_media(self, media_id: str, fetch_media=False, fetch_lazy=True):
        be_media_rec = await self.backend.aget_media(media_id)
        vec_media_rec = await self.vector_store.get_image(media_id)
        vec_metadata:VectorMediaMetadata = vec_media_rec.metadata
        media_rec = Media(
            media=None,
            media_type="VIDEO" if vec_metadata.media_type == "video" else "IMAGE",
            source= vec_metadata.source,
            caption=vec_metadata.description,
            backend_uuid=vec_media_rec.backend_id,
            metadata=vec_metadata,
        )
        media_rec.date = be_media_rec.create_at
        media_rec.filename = be_media_rec.name
        media_rec.thumbnail_filename = be_media_rec.thumbnail
        media_rec.faces = be_media_rec.faces
        media_rec.shots = be_media_rec.events
        media_rec.be_rec = be_media_rec
        media_rec.vector_rec = vec_media_rec
        if fetch_media:
            s3_data = await asyncio.to_thread(
                self._fetch_s3_data, 
                be_media_rec, 
                fetch_lazy=True
            )
            media_rec._media = s3_data.get('media_item', None)
            media_rec.thumbnail = s3_data.get('thumbnail', None)
        return media_rec
        # if media_rec is None:
        #     return None
        # media_item= None
        # if fetch_media:
        #     media_rec = self._populate_s3_data(media_rec, fetch_lazy)            

        # media_rec = Media(
        #     media= media_rec.get("media_item", None),
        #     date=datetime.fromtimestamp(int(media_rec['createAt'])/ 1000),
        #     filename= media_rec['name'],
        #     source= media_rec['source'],
        #     thumbnail=media_rec.get('thumbnail', None),
        #     thumbnail_filename= media_rec.get('thumbnail_filename', None),
        #     metadata= vector_rec.metadata,
        #     url= media_rec['url'],
        # )
        # return media_rec
    # async def get_media(self, media_id: str, fetch_media=False, fetch_lazy=True):
    #     media_rec = self.backend.get_media(media_id)
    #     vector_rec = await self.vector_store.get_image(media_id)
    #     if media_rec is None:
    #         return None
    #     media_item= None
    #     if fetch_media:
    #         media_rec = self._populate_s3_data(media_rec, fetch_lazy)            

    #     media_rec = Media(
    #         media= media_rec.get("media_item", None),
    #         date=datetime.fromtimestamp(int(media_rec['createAt'])/ 1000),
    #         filename= media_rec['name'],
    #         source= media_rec['source'],
    #         thumbnail=media_rec.get('thumbnail', None),
    #         thumbnail_filename= media_rec.get('thumbnail_filename', None),
    #         metadata= vector_rec.metadata,
    #         url= media_rec['url'],
    #     )
    #     return media_rec
        
    
    
    async def aget_images_by_text(self, text: str, filter= None, top_k=5, fetch_backend=True, fetch_images=False):
        similar_images = await self.vector_store.atext_similarity_search(text, filter=filter, top_k=top_k)
        images = []                    
        for si in similar_images:
            image = si.to_dict()
            if fetch_backend:                
                image = self._pupulate_backend_data(image)
            if fetch_images:
                image = self._populate_s3_data(image)
            images.append(image)
        return images
    

    async def search(self, text: str, filter= None, top_k=5, fetch_backend=True, fetch_media=False):
        similar_images = await self.vector_store.atext_similarity_search(text, filter=filter, top_k=top_k)
        # media_list = []                    
        media_list = MediaSearchResultList()
        for vec_media_rec in similar_images:
            # media_rec = si.to_dict()
            media_item= None
            be_media_rec = None
            vec_metadata:VectorMediaMetadata = vec_media_rec.metadata
            media_rec = Media(
                media=None,
                media_type="VIDEO" if vec_metadata.media_type == "video" else "IMAGE",
                source= vec_metadata.source,
                caption=vec_metadata.description,
                backend_uuid=vec_media_rec.backend_id,
                metadata=vec_metadata,
            )
            media_rec.vector_rec = vec_media_rec
            if fetch_backend: 
                be_media_rec = await self.backend.aget_media(vec_media_rec.backend_id)
                media_rec.date = be_media_rec.create_at
                media_rec.filename = be_media_rec.name
                media_rec.thumbnail_filename = be_media_rec.thumbnail
                media_rec.faces = be_media_rec.faces
                media_rec.shots = be_media_rec.events
                media_rec.be_rec = be_media_rec
                
                # if media_rec.type == 'VIDEO':
                #     for ev in be_media_rec.events:
                #         if ev.index == vec_metadata.event_idx:
                #             media_rec.selected_shot = ev
                #             break
                #     else:
                #         raise ImageStoreException(f"event with index {vec_metadata.event_idx} not found in video {be_media_rec.name}")


                # media.events = be_media_rec.events
            
            # if fetch_backend:                
                # media_rec = self._pupulate_backend_data(media_rec)
            if fetch_media:
                s3_data = await asyncio.to_thread(
                    self._fetch_s3_data, 
                    be_media_rec, 
                    fetch_lazy=True
                )
                media_rec._media = s3_data.get('media_item', None)
                media_rec.thumbnail = s3_data.get('thumbnail', None)
            
            media_date = None
            # if media_rec.get('createAt', None):
                # media_date=datetime.fromtimestamp(int(media_rec['createAt'])/ 1000)                        
            # media_rec = Media(
            #     media= media_rec.get("media_item", None),
            #     media_type="VIDEO" if media_rec.metadata.type == "video" else "IMAGE",
            #     caption= media_rec.metadata.de,
            #     # platform_id= ,
            #     date=media_date,
            #     # size= ,
            #     filename= media_rec.get('name', None),
            #     source= media_rec.get('source', None),
            #     thumbnail=media_rec.get('thumbnail', None),
            #     thumbnail_filename= media_rec.get('thumbnail_filename', None),
            #     metadata= media_rec.metadata,
            #     # platform_file_id= ,
            #     # platform_type= ,
            #     # platform_group= ,
                
            #     url= media_rec.get('url', None),
            # )
            media_rec = MediaSearchResult(media_rec, vec_media_rec.score)
            media_list.append(media_rec)
        return media_list
    

    def delete(self, media_id: str):
        self.vector_store.delete(media_id)
        self.backend.delete_media(media_id)


    def delete_media(self, media: Media):
        if type(media) != list:
            media_list = [media]
        else:
            media_list = media
        
        for media in media_list:
            if media.type == 'IMAGE':
                self.vector_store.delete(media.vector_uuid)
            elif media.type == 'VIDEO':
                # self.vector_store.delete(media._media.shots.get_vector_uuids())
                # self.vector_store.delete([e.get_vector_uuids for e in media._media.shots])
                # self.vector_store.delete([e.get_vector_uuids for e in media.shots])
                self.vector_store.delete([e.vector_uuid for e in media.shots])
            self.backend.delete_media(media.backend_uuid)

            self.image_bucket.remove(media.filename)
            self.thumbnail_bucket.remove(media.thumbnail_filename)            
            # delete_s3_obj(self.image_bucket, media.filename)
            # delete_s3_obj(self.thumbnail_bucket, media.thumbnail_filename)


    def delete_all(self, source=None):
        self.vector_store.delete(delete_all=True)
        deleted_media = self.backend.delete_all_media(source)
        for m in deleted_media:
            self.image_bucket.remove(m['name'])
            self.thumbnail_bucket.remove(m['thumbnail'])            
            # delete_s3_obj(self.image_bucket, m['name'])
            # delete_s3_obj(self.thumbnail_bucket, m['thumbnail'])


                
    async def add_user_image(self, image: Image, filename: str, faces: List[Face], metadata: Dict, account_id: str, movie_context: str):
        # image.to_s3(self.image_bucket, filename)
        
        thumbnail_image = image.get_thumbnail()        
        thumbnail_filename = "thumbnail_" + "".join(filename.split('.')[:-1]) + '.png'        
        await asyncio.to_thread(
            self.image_bucket.add,
            image, 
            filename
        )
        await asyncio.to_thread(
            self.thumbnail_bucket.add,
            thumbnail_image, 
            thumbnail_filename
        )

        backend_media_res = await asyncio.to_thread(
            self.backend.add_media,
            name=filename, 
            type='IMAGE',
            source="S3",
            width =image.width,
            height = image.height,
            thumbnail=thumbnail_filename,
            faces=[f.to_dict() for f in faces],
            is_processed=True,
            movie_context=movie_context,
            account_id=account_id,
            metadata=metadata
        )
        return Media(
            media=image,
            thumbnail=thumbnail_image,
            filename=filename,
            thumbnail_filename=thumbnail_filename,
            backend_uuid=backend_media_res['id'],
        )        