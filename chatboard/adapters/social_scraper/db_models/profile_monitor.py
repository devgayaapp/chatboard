from typing import List, Optional
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker
from components.adapters.social_scraper.db_models.helper_models import PlatformType


class ProfileMonitorBase(BaseModel):
    username: str
    profile_id: Optional[str] = None
    platform: PlatformType


class ProfileMonitorCreate(ProfileMonitorBase, DateTimeTracker):
    is_locked: bool = False
    is_valid_profile: bool = True
    fetch_error_counter: int = 0
    post_fetch_error_counter: int = 0
    working_hours: List[int] = list(range(0, 24))  # 0-23 hours in a day


class ProfileMonitor(ProfileMonitorCreate, ObjectIdBase):
    pass


class ProfileQueue(ProfileMonitorBase, ObjectIdBase):
    """
    ProfileQueue is a model for the queue to be processed by the worker.
    This is a model without the datetime fields and the is_locked field
    as it is not needed in the queue and is only needed in the database.
    """
    pass
