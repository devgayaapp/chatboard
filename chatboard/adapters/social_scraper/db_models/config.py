from enum import Enum
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker


class ConfigName(str, Enum):
    DEFAULT = "DEFAULT"
    RABBITMQ_PROFILES_QUEUE = "RABBITMQ_PROFILES_QUEUE"
    RABBITMQ_ENRICHMENT_QUEUE = "RABBITMQ_ENRICHMENT_QUEUE"
    

class ConfigBase(BaseModel):
    config_name: ConfigName
    RABBIT_QUEUE_NAME: str
    RABBIT_PREFETCH_COUNT: int
    PROFILE_TO_QUEUE_PUBLISH_COUNT: int
    PROFILE_TO_QUEUE_PUBLISH_INTERVAL_SECONDS: int
    RABBIT_ASYNC_WORKERS_COUNT: int
    PROFILE_FETCH_ERROR_THRESHOLD: int
    PROFILE_POST_FETCH_ERROR_THRESHOLD: int


class ConfigCreate(ConfigBase, DateTimeTracker):
    pass


class Config(ConfigCreate, ObjectIdBase):
    pass
