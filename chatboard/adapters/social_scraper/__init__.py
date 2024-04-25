# from typing import Optional
# from bson import ObjectId

# from config import SCRAPER_MONGODB_COLLECTION_POSTS
# from components.adapters.social_scraper.db.crud._base import MongoCRUDBase
# from components.adapters.social_scraper.db.mongodb import database
# from components.adapters.social_scraper.db_models.post import Post, PostCreate


# class CRUDServer(MongoCRUDBase[PostCreate, Post]):
#     pass
    

# post = CRUDServer(
#     collection_name=SCRAPER_MONGODB_COLLECTION_POSTS,
#     create_schema=PostCreate,
#     db_schema=Post,
#     db=database.get_database(),
# )
