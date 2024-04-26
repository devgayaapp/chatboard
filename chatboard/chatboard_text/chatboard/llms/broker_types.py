
from enum import Enum
from uuid import uuid4
from pydantic import BaseModel, Field
from typing import Any, Optional
from termcolor import colored

from util.time_utils import timestamp_to_datetime, unix_time



def get_message_id():
    return uuid4().hex
    


class JobStatusEnum(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    RUNNING = "RUNNING"




class AiJobStatusMessage(BaseModel):
    type: str
    job_id: str
    user_id: str
    status: JobStatusEnum
    data: Optional[Any] = None
    timestamp: int = Field(default_factory=unix_time)
    uuid: str = Field(default_factory=get_message_id)
    
    class Config:
        # Allow the timestamp field to be set on initialization
        allow_mutation = True

    def __str__(self) -> str:
        return f"status msg {self.uuid[:6]}:  --{colored(self.type, 'purple')}-- job:{self.job_id[:16]} user:{self.user_id[:6]} {self.status} {timestamp_to_datetime(self.timestamp)}  -> {colored(self.data, 'green')}"
    
    # def __repr__(self) -> str:
    #     return f"{colored(self.type, 'purple')} {self.job_id} {self.user_id} {self.status} {self.timestamp} {self.uuid} -> {colored(self.data, 'green')}"

class AiJobStartMessage(BaseModel):
    user_id: str
    job_id: str
    op: str
    params: Any

class AiSocketMessage(BaseModel):
    op: str
    user_id: str
    params: Optional[Any] = None
    is_processed: bool = False