from pydantic import Field
import pytz
from bson import ObjectId
from datetime import datetime

from components.adapters.social_scraper.db_models._base import ObjectIdBase, utc_now
from components.adapters.social_scraper.db_models.post import InteractionMetrics


class PostTimeSeries(InteractionMetrics):
    post_id: ObjectId

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PostTimeSeriesCreate(PostTimeSeries):
    date_created: datetime = Field(default_factory=utc_now)


class PostTimeSeries(PostTimeSeriesCreate, ObjectIdBase):
    pass
