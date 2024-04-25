from datetime import datetime, timezone
from typing import Optional
import pytz

from bson.objectid import ObjectId
from pydantic import BaseModel, Field, validator
from pydantic_mongo import ObjectIdField


def utc_now():
    return datetime.now(pytz.timezone("UTC"))


class ObjectIdBase(BaseModel):
    id: Optional[ObjectIdField] = Field(alias="_id")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
        }

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["_id"] = ObjectId(d.pop("id"))  # Use '_id' instead of 'id'
        return d

    def model_dump_json(self, **kwargs):
        json_str = super().model_dump_json(**kwargs)
        json_str = json_str.replace(
            '"id":', '"_id":', 1
        )  # Replace the first occurrence of '"id":' with '"_id":'
        return json_str

    @validator("id", pre=True)
    def ensure_objectid(cls, v):
        if isinstance(v, str):
            return ObjectId(v)
        return v

    @validator("*", pre=True)
    def ensure_timezone_aware(cls, value):
        if isinstance(value, datetime):
            if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
                value = value.replace(tzinfo=timezone.utc)
        return value


class DateTimeTracker(BaseModel):
    date_created: datetime = Field(default_factory=utc_now)
    date_last_updated: datetime = Field(default_factory=utc_now)
