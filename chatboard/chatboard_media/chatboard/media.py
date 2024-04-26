from datetime import datetime
from typing import Any, List, Union

from pydantic import BaseModel
from components.image.image import Image
from components.media.media_file_util import gen_thumbnail_filename
from components.video.video import Video
from config import AWS_IMAGE_BUCKET, AWS_THUMBNAIL_BUCKET
import copy
import os



PLATFORM_LOOKUP = {
    "telegram": "1",
    "google_drive": "2",
    "facebook": "3",
    "instagram": "4",
    "twitter": "5",
    "youtube": "6",
    "whatsapp": "7",
    "tiktok": "8",
    "pinterest": "9",
    "reddit": "10",
    "pexels": "11"
}


class Media:


    def __init__(self, 
            media: Union[Video, Image] = None, 
            caption: dict = None, 
            platform_id: str = None, 
            date: int = None, 
            size: int = None, 
            filename: str = None, 
            source: str = None,  
            thumbnail: Image=None,
            thumbnail_filename: str=None,
            metadata: dict=None, 
            platform_file_id: str=None,
            platform_type: str=None,
            platform_group: str=None,
            media_type: str=None,
            url: str=None,
            is_downloaded: str = False,
            embeddings: str = None,
            faces: str = None,
            backend_uuid: str = None,
            pinecone_uuid: str= None,
            pinecone_shots_uuids: List[str]=None,

        ) -> None:
        self._media = media
        if media_type is not None:
            self._type = media_type
        else:
            if type(media) == Video or str(type(media)) == str(Video):
                self._type = 'VIDEO'
            elif type(media) == Image or str(type(media)) == str(Image):
                self._type = 'IMAGE'
            else:
                self._type = 'UNKNOWN'
        self.caption = caption
        self.platform_id = platform_id
        if type(date) == str:
            self.date = datetime.fromisoformat(date)
        else:
            self.date = date
        self.size = size
        self.filename = filename
        if thumbnail_filename is None and thumbnail is not None:
            self.thumbnail_filename = gen_thumbnail_filename(filename)
        else:
            self.thumbnail_filename = thumbnail_filename
        self.thumbnail = thumbnail
        self._is_downloaded = is_downloaded
        self.source = source
        self.platform_file_id = platform_file_id
        self.platform_type = platform_type
        self.metadata = metadata
        self.be_rec = None
        self.vector_rec = None
        self.embeddings = embeddings
        self.faces = faces
        self.platform_group = platform_group
        self.backend_uuid = backend_uuid
        self._pinecone_uuid = pinecone_uuid
        self._pinecone_shots_uuids = pinecone_shots_uuids
        self.url = url
        self._shots = None
        self.selected_shot = None

    @property
    def height(self):
        if self._media is not None:
            return self._media.height
        if self.be_rec is not None:
            return self.be_rec.height
    
    @property
    def width(self):
        if self._media is not None:
            return self._media.width
        if self.be_rec is not None:
            return self.be_rec.width

    def to_json(self):
        metadata = self.metadata
        if isinstance(self.metadata, BaseModel):
            metadata = self.metadata.dict()
        return {
            'backend_uuid': self.backend_uuid,
            'caption': self.caption,
            'platform_id': self.platform_id,
            'date': self.date.isoformat() if self.date else None,
            'size': self.size,
            'filename': self.filename,
            'thumbnail_filename': self.thumbnail_filename,
            'source': self.source,
            'platform_file_id': self.platform_file_id,
            'platform_type': self.platform_type,
            'metadata': metadata,
            'platform_group': self.platform_group,
            'media_type': self._type,
        }
    
    def deep_copy(self):
        if self.type == 'VIDEO' and self._media._video_clip is not None:
            tmp_video_clip = self._media._video_clip
            self._media._video_clip = None
            media_copy = copy.deepcopy(self)
            self._media._video_clip = tmp_video_clip
            return media_copy
        return copy.deepcopy(self)
    
    def copy(self):
        media_copy = Media(
            media=self.media,
            caption=self.caption,
            platform_id=self.platform_id,
            date=self.date,
            size=self.size,
            filename=self.filename,
            source=self.source,
            thumbnail=self.thumbnail,
            thumbnail_filename=self.thumbnail_filename,
            metadata=self.metadata,
            platform_file_id=self.platform_file_id,
            platform_type=self.platform_type,
            platform_group=self.platform_group,
            media_type=self.media_type,
            url=self.url,
            is_downloaded=self._is_downloaded,
            embeddings=self.embeddings,
            faces=self.faces,
            backend_uuid=self.backend_uuid,
            pinecone_uuid=self._pinecone_uuid,
            pinecone_shots_uuids=self._pinecone_shots_uuids

        )

        return media_copy
    
    @staticmethod
    def from_json(json):
        return Media(
            caption=json['caption'],
            platform_id=json['platform_id'],
            date=json['date'],
            size=json['size'],
            filename=json['filename'],
            thumbnail_filename=json['thumbnail_filename'],
            source=json['source'],
            platform_file_id=json['platform_file_id'],
            platform_type=json['platform_type'],
            metadata=json['metadata'],
            platform_group=json['platform_group'],
            media_type=json['media_type'],
        )

    def _repr_png_(self):
        print(self.type)
        if self.thumbnail:
            return self.thumbnail._repr_png_()
        
    def gen_unique_id(self):
        # uuid = f"{self.platform_type}_{self.platform_id}_{self.platform_file_id}"        
        uuid = f"{PLATFORM_LOOKUP[self.platform_type]}_{self.platform_id}"
        if self.platform_file_id:
            uuid += f"_f{self.platform_file_id}"
        return uuid
        
    
    @property
    def vector_uuid(self):
        if self._type == 'VIDEO':
            return self._pinecone_uuid
        else:
            return self._pinecone_uuid
    
    @property
    def shots(self):
        if self._media is not None:
            return self._media.shots
        return self._shots
    
    @shots.setter
    def shots(self, shots):
        self._shots = shots

    def get_shot_by_index(self, index):
        # if self._media is not None:
        #     return self._media.get_shot_by_index(index)
        if self._shots is not None:
            return [s for s in self._shots if s.index == index][0]

    @property
    def type(self):
        return self._type

    @property
    def is_saved(self):
        if self.backend_uuid is not None and self.vector_uuid is not None:
            return True
        else:
            return False
        
    def populate(self):
        if self._media is None:
            self.thumbnail = Image.from_s3(AWS_THUMBNAIL_BUCKET, self.filename)
            if self.type == 'IMAGE':
                self._media = Image.from_s3(AWS_IMAGE_BUCKET, self.filename)
        return self


    def to_s3(self, bucket=AWS_IMAGE_BUCKET, thumbnail_bucket=AWS_THUMBNAIL_BUCKET):
        if self._media is not None:
            self._media.to_s3(bucket, self.filename)
        if self.thumbnail is not None:
            self.thumbnail.to_s3(thumbnail_bucket, self.thumbnail_filename or self.filename)
        return self
    
    def to_file(self, path):
        if self._media is not None:
            self._media.to_file(path)
        
    
    def delete(self):
        if self.type == 'VIDEO':
            self._media.delete()



    @staticmethod
    def grid(media_list, cols=4):
        return Image.grid([m.thumbnail for m in media_list], cols = cols)