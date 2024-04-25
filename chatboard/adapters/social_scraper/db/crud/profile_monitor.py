from datetime import datetime
from typing import List
from bson import ObjectId
import pytz

from app.core.config import COLLECTION_PROFILES_MONITOR
from app.infrastructure.db.crud._base import MongoCRUDBase
from app.infrastructure.db.mongodb import database
from app.models.profile_monitor import (
    ProfileMonitorCreate,
    ProfileMonitor,
    ProfileQueue,
)


class CRUDProfileMonitor(MongoCRUDBase[ProfileMonitorCreate, ProfileMonitor]):
    """CRUD operations for ProfileMonitor model."""

    async def get_profiles_for_queue(
        self,
        updated_before: datetime,
        limit: int = 1000,
        fetch_error_threshold: int = 5,
        post_fetch_error_threshold: int = 10,
    ) -> List[ProfileQueue]:
        """Get profiles to be processed in the queue."""
        current_hour = datetime.now(pytz.timezone("UTC")).hour
        docs = (
            await self.collection.find(
                {
                    "date_last_updated": {"$lte": updated_before},
                    "working_hours": current_hour,
                    "is_locked": False,
                    "is_valid_profile": True,
                    "fetch_error_counter": {"$lt": fetch_error_threshold},
                    "post_fetch_error_counter": {"$lt": post_fetch_error_threshold},
                },
                {
                    "username": 1,
                    "profile_id": 1,
                    "platform": 1,
                    "_id": 1,
                },  # annotation keys of ProfileQueue
            )
            .limit(limit)
            .to_list(None)
        )
        return [ProfileQueue(**doc) for doc in docs]

    async def lock_profiles(self, profile_ids: List[ObjectId]):
        """Lock profiles to prevent other workers from processing them."""
        await self.update_many(
            {"_id": {"$in": profile_ids}},
            {"$set": {"is_locked": True}},
        )

    async def release_profile(self, profile_id: ObjectId):
        """Release profile after processing."""
        await self.update_one(
            {"_id": profile_id},
            {
                "$set": {
                    "is_locked": False,
                    "fetch_error_counter": 0,
                    "date_last_updated": datetime.now(pytz.timezone("UTC")),
                }
            },
        )

    async def mark_as_deleted(self, profile_id: ObjectId):
        """Mark profile as deleted."""
        await self.update_one(
            {"_id": profile_id},
            {
                "$set": {
                    "is_valid_profile": False,
                    "is_locked": False,
                    "date_last_updated": datetime.now(pytz.timezone("UTC")),
                }
            },
        )

    async def inc_fetch_error_counter(self, profile_id: ObjectId):
        """Increment fetch error counter."""
        await self.update_one(
            {"_id": profile_id},
            {
                "$set": {
                    "is_locked": False,
                    "date_last_updated": datetime.now(pytz.timezone("UTC")),
                },
                "$inc": {"fetch_error_counter": 1},
            },
        )

    async def inc_post_fetch_error_counter(self, profile_id: ObjectId):
        """Increment fetch error counter."""
        await self.update_one(
            {"_id": profile_id},
            {
                "$set": {
                    "is_locked": False,
                    "date_last_updated": datetime.now(pytz.timezone("UTC")),
                },
                "$inc": {"post_fetch_error_counter": 1},
            },
        )

    async def upsert_monitoring_profile(self, profile: ProfileMonitorCreate):
        """Update or insert new monitoring profile."""
        set_on_insert_dict = profile.model_dump()

        await self.update_one(
            filter_dict={
                "username": profile.username,
                "platform": profile.platform,
            },
            update_dict={
                "$setOnInsert": set_on_insert_dict,
            },
            upsert=True,
        )


profile_monitor = CRUDProfileMonitor(
    collection_name=COLLECTION_PROFILES_MONITOR,
    create_schema=ProfileMonitorCreate,
    db_schema=ProfileMonitor,
    db=database.get_database(),
)
