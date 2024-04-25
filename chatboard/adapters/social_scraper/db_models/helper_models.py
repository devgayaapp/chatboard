from enum import Enum
from typing import Optional
from pydantic import BaseModel


class PlatformType(str, Enum):
    FACEBOOK: str = "facebook"
    INSTAGRAM: str = "instagram"
    TWITTER: str = "twitter"
    YOUTUBE: str = "youtube"
    TIKTOK: str = "tiktok"
    LINKEDIN: str = "linkedin"
    REDDIT: str = "reddit"


