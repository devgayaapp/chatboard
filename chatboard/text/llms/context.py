
from components.etl.logger import Logger


class Context:


    def __init__(self, task_name, job_id, user_id, job_start_msg):
        self.task_name = task_name
        self.job_id = job_id
        self.user_id = user_id
        self.logger = Logger(task_name, job_id, user_id)
        self.job_start_msg = job_start_msg


    async def init(self):
        await self.logger.init()

    async def close(self):
        await self.logger.close()