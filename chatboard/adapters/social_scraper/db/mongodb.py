import logging

from motor.motor_asyncio import AsyncIOMotorClient

from config import SCRAPER_MONGODB_URL, SCRAPER_MONGODB_DB


class DataBase:
    def __init__(self, server_url: str, db_name: str):
        self.client: AsyncIOMotorClient = None
        self.server_url = server_url
        self.db_name = db_name

    async def connect(self):
        # logging.info(f"Connecting to MongoDB server at {self.db_name}")
        self.client = AsyncIOMotorClient(self.server_url)
        # logging.info("Connected")
        logging.info(f"Connecting to MongoDB server at {self.db_name}")

    async def close(self):
        # logging.info(f"Closing MongoDB connection to {self.db_name}")
        if self.client:
            self.client.close()
            # logging.info("Closed")
            logging.info(f"Closing MongoDB connection to {self.db_name}")

    def get_database(self) -> AsyncIOMotorClient:
        if not self.client:
            raise Exception(
                f"You must connect to the database before accessing it "
                f"(server_url={self.db_name})"
            )
        return self.client[self.db_name]


# database = DataBase(server_url=SCRAPER_MONGODB_URL, db_name=SCRAPER_MONGODB_DB)
