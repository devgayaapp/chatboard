from datetime import datetime, timedelta
from typing import List, Optional
from bson import ObjectId

import pytz

from config import SCRAPER_MONGODB_COLLECTION_POSTS
from components.adapters.social_scraper.db.crud._base import MongoCRUDBase
# from components.adapters.social_scraper.db.mongodb import database
from components.adapters.social_scraper.db_models.post import Post, PostCreate


class CRUDPost(MongoCRUDBase[PostCreate, Post]):
    async def get_posts(self, after_date: datetime) -> List[Post]:
        """Get posts created after a certain date."""
        return await self.get_many(
            filter={
                "date_post_created": {"$gte": after_date},
            },
        )

    async def upsert_post(self, post: PostCreate):
        """Update or insert new post."""
        set_on_insert_dict = post.model_dump()

        interaction_metrics = set_on_insert_dict.pop("interaction_metrics")
        date_last_updated = set_on_insert_dict.pop("date_last_updated")
        thumbnail_source = set_on_insert_dict.pop("thumbnail_source")
        display_url = set_on_insert_dict.pop("display_url")
        video_url = set_on_insert_dict.pop("video_url")

        await self.update_one(
            filter_dict={"post_id": post.post_id},
            update_dict={
                "$set": {
                    "interaction_metrics": interaction_metrics,
                    "date_last_updated": date_last_updated,
                    "thumbnail_source": thumbnail_source,
                    "display_url": display_url,
                    "video_url": video_url,
                },
                "$setOnInsert": set_on_insert_dict,
            },
            upsert=True,
        )

    async def get_posts_for_ai_enrichment_queue(
        self,
        limit: int = 1000,
        ai_enrichment_error_threshold: int = 5,
        max_post_age_seconds: int = 600,
    ) -> List[Post]:
        newer_than_time = datetime.now(pytz.timezone("UTC")) - timedelta(
            seconds=max_post_age_seconds
        )
        docs = (
            await self.collection.find(
                {
                    "date_created": {"$gte": newer_than_time},
                    "caption": {"$ne": None},
                    "stance": None,
                    "ai_enrichment_counter": {"$lt": ai_enrichment_error_threshold},
                },
                # {
                #     "caption": 1,
                #     "_id": 1,
                # },
            )
            .limit(limit)
            .to_list(None)
        )
        return [Post(**doc) for doc in docs]

    async def update_post_ai_enrichment_status(
        self,
        post_db_id: ObjectId,
        main_theme: Optional[str],
        topics: Optional[List[str]],
        stance: Optional[str],
        language: Optional[str],
    ):
        await self.update_one(
            filter_dict={"_id": post_db_id},
            update_dict={
                "$set": {
                    "main_theme": main_theme,
                    "topics": topics,
                    "stance": stance,
                    "language": language,
                },
                "$inc": {"ai_enrichment_counter": 1},
            },
        )


# post = CRUDPost(
#     collection_name=SCRAPER_MONGODB_COLLECTION_POSTS,
#     create_schema=PostCreate,
#     db_schema=Post,
#     db=database.get_database(),
# )
