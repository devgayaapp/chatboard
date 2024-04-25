from config import COLLECTION_POSTS_TIME_SERIES
from components.adapters.social_scraper.db.crud._base import MongoCRUDBase
from components.adapters.social_scraper.db.mongodb import database
from components.adapters.social_scraper.db_models.post_time_series import PostTimeSeriesCreate, PostTimeSeries


class CRUDPostTimeSeries(MongoCRUDBase[PostTimeSeriesCreate, PostTimeSeries]):
    pass


post_time_series = CRUDPostTimeSeries(
    collection_name=COLLECTION_POSTS_TIME_SERIES,
    create_schema=PostTimeSeriesCreate,
    db_schema=PostTimeSeries,
    db=database.get_database(),
)
