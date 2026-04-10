from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGO_URI, DATABASE_NAME
from loguru import logger
import asyncio

class MongoDBHandler:
    def __init__(self, uri: str, db_name: str):
        self.client = AsyncIOMotorClient(uri)
        self.db = self.client[db_name]
        logger.info(f"Connected to MongoDB at {uri}")

    async def save_response(self, user_id: str, response_data: dict):
        """
        Saves the response data to a collection named after the user_id.
        This operation is asynchronous and should not block the main API response.
        """
        try:
            collection = self.db[user_id]
            result = await collection.insert_one(response_data)
            logger.info(f"Successfully saved response to collection {user_id} with id {result.inserted_id}")
            return result.inserted_id
        except Exception as e:
            logger.error(f"Failed to save response to MongoDB: {e}")
            return None

# Global instance
mongo_handler = MongoDBHandler(MONGO_URI, DATABASE_NAME)
