


from datetime import datetime
from typing import List, Union

from chatboard.text.llms.views import BaseModel, Field

from chatboard.text.llms.chat_prompt import ChatPrompt
from chatboard.text.llms.conversation import HumanMessage, SystemMessage, AIMessage


class HistoryMessage:

    def __init__(self, view_name, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any, date: any, view_cls=None) -> None:
        self.inputs = inputs
        self.output = output
        self.msgs = msgs
        self.view_name = view_name
        self.view_cls = view_cls
        self.date = date

    def get_input_message(self):
        return self.msgs[-1]
    
    def get_output_message(self):
        return self.output


class History:

    def __init__(self) -> None:
        self.history = []

    def add(self, view_name, view_cls, msgs: List[HumanMessage | SystemMessage | AIMessage], inputs: any, output: any):
        self.history.append(HistoryMessage(
                view_name=view_name, 
                view_cls=view_cls, 
                inputs=inputs, 
                output=output,
                date=datetime.now(),
                msgs=msgs
            ))


    def get(self, view_name=None, top_k=1):
        if view_name is None:
            return self.history[:top_k]
        hist_msgs = [msg for msg in self.history if msg.view_name == view_name]
        return hist_msgs[:top_k]
    
    async def get_messages(self, view_name=None, top_k=1):
        hist_msgs = self.get(view_name, top_k)
        history_msgs = []
        for msg in hist_msgs:
            history_msgs.append(msg.get_input_message())
            history_msgs.append(msg.get_output_message())
        return history_msgs
        
    async def __call__(self, view_name=None, top_k=1):
        return await self.get_messages(view_name, top_k)


class Context(BaseModel):
    key: str | int

    curr_prompt: ChatPrompt = None
    message: str = None
    history: History = Field(default_factory=History)

    async def init_state(self):
        raise NotImplementedError
    
    class Config:
        arbitrary_types_allowed = True
