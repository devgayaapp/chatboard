import asyncio
from typing import Union
from aiokafka import AIOKafkaProducer
from pydantic import BaseModel
from components.etl.broker_types import JobStatusEnum, AiJobStatusMessage
from config import KAFKA_BOOTSTRAP_SERVERS
import traceback

import logging

# etl_logger = logging.getLogger("etl_logger")

class Logger:

    def __init__(self, task_name, job_id, user_id) -> None:
        self.task_name = task_name
        self.job_id = job_id
        self.user_id = user_id
        self.loop = asyncio.get_running_loop()
        self.producer = AIOKafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, 
            loop=self.loop
        )

    async def emit(self, message: Union[str,dict], status: JobStatusEnum):
        message = AiJobStatusMessage(
            type=self.task_name,
            job_id=self.job_id, 
            user_id=self.user_id, 
            status=status, 
            data=message
        )
        await self.send_one(self.user_id, message)


    async def log(self, message: Union[str,dict]):
        logging.info(message)
        await self.emit(message, JobStatusEnum.RUNNING)

    async def error(self, message: Union[str, dict, Exception]):
        logging.error(message)
        if isinstance(message, Exception):
            traceback.print_exc()
        await self.emit(str(message), JobStatusEnum.FAILED)


    async def send_one(self, user_id, message: BaseModel):
        await self.producer.send_and_wait(f'user_output_topic_{user_id}', str.encode(message.json()))


    async def init(self):
        await self.producer.start()

    async def close(self):
        await self.producer.stop()


    