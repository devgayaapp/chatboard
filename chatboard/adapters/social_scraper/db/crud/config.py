from typing import Optional
from app.core.config import COLLECTION_CONFIGS
from app.infrastructure.db.mongodb import database
from app.infrastructure.db.crud._base import MongoCRUDBase
from app.models.config import ConfigCreate, Config, ConfigName


class CRUDConfig(MongoCRUDBase[ConfigCreate, Config]):
    
    async def get_config(self, config_name: ConfigName) -> Optional[Config]:
        return await self.get_one({"config_name": config_name})


config = CRUDConfig(
    collection_name=COLLECTION_CONFIGS,
    create_schema=ConfigCreate,
    db_schema=Config,
    db=database.get_database(),
)
