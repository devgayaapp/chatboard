import logging
from typing import Generic, TypeVar, Type, List, Optional

from bson.objectid import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

CreateSchemaType = TypeVar("CreateSchemaType")
InDBSchemaType = TypeVar("InDBSchemaType")


class MongoCRUDBase(Generic[CreateSchemaType, InDBSchemaType]):
    """
    Base class for MongoDB CRUD operations.

    Args:
        collection_name: The name of the collection in the MongoDB database.
        create_schema: The type of the schema used for creating objects.
        db_schema: The type of the schema used for objects stored in the database.
        db: The instance of AsyncIOMotorClient representing the database connection.

    Attributes:
        db: The instance of AsyncIOMotorClient representing the database connection.
        collection: The collection in the MongoDB database.
        create_schema: The type of the schema used for creating objects.
        db_schema: The type of the schema used for objects stored in the database.
        collection_name: The name of the collection in the MongoDB database.
    """

    def __init__(
        self,
        collection_name: str,
        create_schema: Type[CreateSchemaType],
        db_schema: Type[InDBSchemaType],
        db: AsyncIOMotorClient,
    ):
        self.db = db
        self.collection = db[collection_name]
        self.create_schema = create_schema
        self.db_schema = db_schema
        self.collection_name = collection_name

    async def add(self, obj: CreateSchemaType) -> ObjectId:
        """Adds a new object to the collection."""
        _id = (await self.collection.insert_one(obj.model_dump())).inserted_id
        if not _id:
            logger.info(f"Error inserting document: {obj}")
        return _id

    async def add_many(self, objects: List[CreateSchemaType]):
        """Adds multiple new objects to the collection."""
        await self.collection.insert_many([obj.browser() for obj in objects])

    async def get_one(
        self, filter_dict: dict, sort: List[tuple] = None
    ) -> Optional[InDBSchemaType]:
        """Retrieves a single object from the collection."""
        if sort:
            doc = await self.collection.find_one(filter_dict, sort=sort)
        else:
            doc = await self.collection.find_one(filter_dict)

        if doc is None:
            return None
            # raise ValueError(f"No document found with filter: {filter_dict}")
        return self.db_schema(**doc)

    async def get_many(
        self, filter_dict: dict, sort: List = None, limit: int = None
    ) -> List[InDBSchemaType]:
        """Retrieves multiple objects from the collection."""
        if sort:
            return [
                self.db_schema(**doc)
                for doc in await self.collection.find(filter_dict, sort=sort).to_list(
                    limit
                )
            ]
        return [
            self.db_schema(**doc)
            for doc in await self.collection.find(filter_dict).to_list(limit)
        ]

    async def get_all(self) -> List[InDBSchemaType]:
        """Retrieves all objects from the collection."""
        return [
            self.db_schema(**doc) for doc in await self.collection.find().to_list(None)
        ]

    async def get_docs_count(self, filter_dict: dict) -> int:
        """Returns the number of documents matching the filter."""
        return await self.collection.count_documents(filter_dict)

    async def update_one(
        self, filter_dict: dict, update_dict: dict, upsert: bool = False
    ) -> None:
        """Updates a single object in the collection."""
        result = await self.collection.update_one(
            filter_dict, update_dict, upsert=upsert
        )
        if result.modified_count != 1:
            # raise ValueError(f"No document updated with filter: {filter_dict}")
            pass

    async def update_many(self, filter_dict: dict, update_dict: dict) -> None:
        """Updates a multiple objects in the collection."""
        result = await self.collection.update_many(filter_dict, update_dict)
        if result.modified_count != 1:
            # raise ValueError(f"No document updated with filter: {filter_dict}")
            pass

    async def find_one_and_update(
        self, query: dict, update: dict, sort: list = None
    ) -> Optional[InDBSchemaType]:
        """Finds an object in the collection, updates it and returns the updated object."""
        options = {"return_document": ReturnDocument.AFTER}
        if sort:
            options["sort"] = sort
        doc = await self.collection.find_one_and_update(query, update, **options)
        if doc is None:
            return None
        return self.db_schema(**doc)

    async def get_random_sample(
        self, filter_dict: dict, sample_size: int
    ) -> List[InDBSchemaType]:
        """Retrieves a random sample of objects from the collection."""
        return [
            self.db_schema(**doc)
            for doc in await self.collection.aggregate(
                [{"$match": filter_dict}, {"$sample": {"size": sample_size}}]
            ).to_list(None)
        ]
