from app.core.config import COLLECTION_PROFILES
from app.infrastructure.db.crud._base import MongoCRUDBase
from app.infrastructure.db.mongodb import database
from app.models.profile import ProfileCreate, Profile


class CRUDProfile(MongoCRUDBase[ProfileCreate, Profile]):
    """
    CRUD operations for Profile model.
    Inherits from MongoCRUDBase and specifies ProfileCreate and Profile models.
    """

    async def upsert_profile(self, profile: ProfileCreate):
        """Update or insert new profile."""
        set_on_insert_dict = profile.model_dump()

        profile_metrics = set_on_insert_dict.pop("profile_metrics")
        date_last_updated = set_on_insert_dict.pop("date_last_updated")

        await self.update_one(
            filter_dict={
                "username": profile.username,
                "platform": profile.platform,
            },
            update_dict={
                "$set": {
                    "profile_metrics": profile_metrics,
                    "date_last_updated": date_last_updated,
                },
                "$setOnInsert": set_on_insert_dict,
            },
            upsert=True,
        )


profile = CRUDProfile(
    collection_name=COLLECTION_PROFILES,
    create_schema=ProfileCreate,
    db_schema=Profile,
    db=database.get_database(),
)
