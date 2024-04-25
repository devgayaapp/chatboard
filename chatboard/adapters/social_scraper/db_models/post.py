from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker
from components.adapters.social_scraper.db_models.helper_models import PlatformType


class PostType(str, Enum):
    POST = "post"
    REEL = "reel"


class MediaType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    TEXT = "text"


class PostInteractionMetrics(BaseModel):
    like_count: Optional[int] = 0
    love_count: Optional[int] = 0
    care_count: Optional[int] = 0
    haha_count: Optional[int] = 0
    wow_count: Optional[int] = 0
    sad_count: Optional[int] = 0
    angry_count: Optional[int] = 0
    comment_count: Optional[int] = 0
    watch_count: Optional[int] = 0
    share_count: Optional[int] = 0


class PostBase(BaseModel):
    post_id: str
    post_shortcode: str
    owner_id: str
    owner_username: str
    post_url: str
    platform: PlatformType
    post_type: PostType
    media_type: MediaType

    date_post_created: datetime
    comments_disabled: Optional[bool] = None

    interaction_metrics: PostInteractionMetrics = PostInteractionMetrics()

    caption: Optional[str] = None
    accessibility_caption: Optional[str] = None
    thumbnail_source: Optional[str] = None
    display_url: Optional[str] = None
    video_url: Optional[str] = None

    main_theme: Optional[str] = None
    topics: Optional[List[str]] = None
    stance: Optional[str] = None
    language: Optional[str] = None
    ai_enrichment_counter: int = 0


class PostCreate(PostBase, DateTimeTracker):
    pass


class Post(PostCreate, ObjectIdBase):
    pass
