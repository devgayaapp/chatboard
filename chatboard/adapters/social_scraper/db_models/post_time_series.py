from pydantic import Field
from datetime import datetime

from components.adapters.social_scraper.db_models._base import ObjectIdBase, utc_now
from components.adapters.social_scraper.db_models.helper_models import PlatformType
from components.adapters.social_scraper.db_models.post import PostInteractionMetrics


class PostTimeSeriesBase(PostInteractionMetrics):
    post_shortcode: str
    platform: PlatformType


class PostTimeSeriesCreate(PostTimeSeriesBase):
    date_created: datetime = Field(default_factory=utc_now)


class PostTimeSeries(PostTimeSeriesCreate, ObjectIdBase):
    pass
