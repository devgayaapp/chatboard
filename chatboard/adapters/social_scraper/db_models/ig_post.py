from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker
from components.adapters.social_scraper.db_models.helper_models import PlatformType
from components.adapters.social_scraper.db_models.ai_post_enrichment import PostAiEnrichment


class MediaType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    TEXT = "text"


class PostInteractionMetrics(BaseModel):
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    view_count: Optional[int] = None


class PostType(str, Enum):
    POST = "post"
    REEL = "reel"


class PostBase(BaseModel):
    platform: PlatformType
    post_id: str

    owner_id: str
    owner_username: str

    text: Optional[str] = None

    post_url: str
    display_url: Optional[str] = None
    thumbnail_source: Optional[str] = None
    video_url: Optional[str] = None
    media_type: MediaType
    date_post_created: datetime
    interaction_metrics: PostInteractionMetrics = PostInteractionMetrics()

    # post_media_id: str
    post_type: PostType
    accessibility_caption: Optional[str] = None
    comments_disabled: Optional[bool] = None

    ai_enrichment: PostAiEnrichment = PostAiEnrichment()


class PostCreate(PostBase, DateTimeTracker):
    pass


class Post(PostCreate, ObjectIdBase):
    pass
