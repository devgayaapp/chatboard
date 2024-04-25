from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker
from components.adapters.social_scraper.db_models.helper_models import PlatformType
from components.adapters.social_scraper.db_models.ai_post_enrichment import PostAiEnrichment


class MediaType(str, Enum):
    PHOTO = "photo"
    VIDEO = "video"
    TEXT = "text"
    GIF = "gif"
    POLL = "poll"


class PostInteractionMetrics(BaseModel):
    bookmark_count: Optional[int] = None
    favorite_count: Optional[int] = None
    quote_count: Optional[int] = None
    reply_count: Optional[int] = None
    retweet_count: Optional[int] = None


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

    video_urls_list: Optional[List[str]] = None
    photo_urls_list: Optional[List[str]] = None
    attached_urls_list: Optional[List[str]] = None

    retweeted: Optional[bool] = None
    is_quote_status: Optional[bool] = None
    mentioned_in_tweets: Optional[List[str]] = []
    is_note: Optional[bool] = None
    twitter_lang: Optional[str] = None

    attached_url: Optional[str] = None
    user_mentions: Optional[List[str]] = None
    hashtags: Optional[List[str]] = None

    ai_enrichment: PostAiEnrichment = PostAiEnrichment()


class PostCreate(PostBase, DateTimeTracker):
    pass


class Post(PostCreate, ObjectIdBase):
    pass
