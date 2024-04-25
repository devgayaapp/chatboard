from typing import Optional
from pydantic import BaseModel

from components.adapters.social_scraper.db_models._base import ObjectIdBase, DateTimeTracker
from components.adapters.social_scraper.db_models.helper_models import PlatformType


class BusinessAddress(BaseModel):
    city_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    street_address: Optional[str] = None
    zip_code: Optional[str] = None


class ProfileMetrics(BaseModel):
    posts_count: int = 0
    follower_count: int = 0
    following_count: int = 0


class ProfileBase(BaseModel):
    username: str
    profile_id: Optional[str] = None
    platform: PlatformType

    full_name: Optional[str] = None
    biography: Optional[str] = None

    business_category: Optional[str] = None
    business_contact_method: Optional[str] = None
    category_enum: Optional[str] = None
    category_name: Optional[str] = None
    profile_pic_url: Optional[str] = None

    address: Optional[BusinessAddress] = BusinessAddress()
    profile_metrics: Optional[ProfileMetrics] = ProfileMetrics()


class ProfileCreate(ProfileBase, DateTimeTracker):
    pass


class Profile(ProfileCreate, ObjectIdBase):
    pass
