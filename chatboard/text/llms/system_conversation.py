




import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, validator
from .rag_manager import RagValue, RagVector, RagVectorSpace
from langsmith import Client
# from ...common.vectors.embeddings.text_embeddings import DenseEmbeddings
from ..vectors.embeddings.text_embeddings import DenseEmbeddings


class BaseMessage(BaseModel):
    content: str
    example: Optional[bool] = False
    is_output: Optional[bool] = False

    def to_openai(self):
        return {
            "role": self.role,
            "content": self.content,
        }


class SystemMessage(BaseMessage):
    role: Optional[str] = "system"


class HumanMessage(BaseMessage):
    role: Optional[str] = "user"


class AIMessage(BaseMessage):
    role: Optional[str] = "assistant"
    tool_calls: Optional[List[BaseModel]] = None
    
    # tools: Optional[List[BaseModel]] = None


def from_langchain_message(message):
    if message.type == "system":
        return SystemMessage(content=message.content)
    elif message.type == "human":
        return HumanMessage(content=message.content)
    elif message.type == "ai":
        return AIMessage(content=message.content, tool_calls=message.tool_calls)
    else:
        raise ValueError(f"Invalid role: {message.role}")
    

def to_langsmith_message(message, is_output=False):
    if message["role"] == "system":
        return SystemMessage(content=message["content"])
    elif message["role"] == "user":
        return HumanMessage(content=message["content"])
    elif message["role"] == "assistant":
        return AIMessage(
            content=message["content"], 
            tool_calls=message.get("tool_calls", None), 
            is_output=is_output
        )
    else:
        raise ValueError(f"Invalid role: {message['role']}")


class ConversationRagMetadata(BaseModel):
    """inputs can be a list of messages or a single message"""
    inputs: str
    output: str


    # @property
    # def key(self):
    #     return f"""inpute_messages: {[m['content'] for m in self.input if m['is_output'] == False]}"""

    

class Conversation:


    def __init__(self, messages: List[Union[SystemMessage, HumanMessage, AIMessage]]=None, system_metadata=None, user_metadata=None, conv_id=None) -> None:
        self.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = messages or []
        self.system_metadata = system_metadata
        self.user_metadata = user_metadata
        self.id = conv_id


    @staticmethod
    def from_json(json_messages):
        return Conversation(messages=[to_langsmith_message(m) for m in json_messages])
        
    @property
    def response(self):
        return self.messages[-1].content

    def append(self, message):
        self.messages.append(message)

    def get_messages(self, top_k=5):
        return self.messages[:-top_k]
    
    def trim_memory(self, top_k=5):
        if self.messages[0].role == "system" and len(self.messages) > top_k:
            messages = [self.messages[0]] + self.messages[-top_k:]
        else:
            messages = self.messages[-top_k:]
        return Conversation(messages=messages, system_metadata=self.system_metadata, user_metadata=self.user_metadata, conv_id=self.id)
    

    def has_system_prompt(self):
        if len(self.messages) == 0:
            return False
        return self.messages[0].role == "system"
    

    def __getitem__(self, index):
        return self.messages[index]
    
    def __len__(self):
        return len(self.messages)
    
    def to_openai(self):            
        return [m.to_openai() for m in self.messages]
    

    def get_metadata(self):
        metadata = {}
        if self.user_metadata is not None:
            metadata['user_prompt'] = self.user_metadata['prompt']          
            metadata['user_commit'] = self.user_metadata['commit']
        if self.system_metadata is not None:
            metadata['system_prompt'] = self.system_metadata['prompt']
            metadata['system_commit'] = self.system_metadata['commit']
        return metadata
    
    def copy(self):
        new_convo = Conversation()
        new_convo.messages = [m.copy() for m in self.messages]
        for msg in new_convo.messages:
            if msg.role == "assistant":
                msg.is_output = False
        return new_convo
    

    def message_html(self, message):
        
        message_html = message.content.replace("\n", "<br>")
        tools_html = []
        is_output = ""
        if message.role == "user":
            label = """<span style="padding: 3px; background: blue; color: white;" >user</span>"""
        elif message.role == "system":
            label = """<span style="padding: 3px; background: green; color: white;" >system</span>"""
        elif message.role == "assistant":
            label = """<span style="padding: 3px; background: red; color: white;" >assistant</span>"""
            is_output = """<span style="padding: 3px; background: orange; color: white;" >output</span>"""
            if message.tool_calls:
                for tool in message.tool_calls:
                    tools_html.append(f"""
                    <div style="border: 1px solid; margin: 5px;">
                    <span style="padding: 3px; background: black; color: white;" >tool: {type(tool).__name__}</span>
                    <p style="width: 600px; font-size: 14px;">
                    {tool}
                    </p>
                    </div>
                    """)
        else:
            label = """<span style="padding: 3px; background: gray; color: white;" >unknown</span>"""
        html = f"""
<div style="border: 1px solid;">
{label} {is_output}
<p style="width: 600px; font-size: 14px;">
             {message_html}
</p>
"""     
        if tools_html:
            for tool_html in tools_html:
                html += tool_html
        html += "</div>"
        return html

    def get_html(self, show_system=True):
        output_html = ""
        for msg in self.messages:
            if show_system == False and msg.role == "system":
                continue
            output_html += self.message_html(msg)
        return output_html
        
    def _repr_html_(self):
        return self.get_html(show_system=False)
 

    # @staticmethod
    # def from_langsmith_record(record, is_agent_tool=False):
    #     input_messages = [m['data'] for m in record.inputs['input']]
    #     if is_agent_tool:
    #         output_message = input_messages[-1]
    #         input_messages = input_messages[:-1]
    #         tool_calls = record.outputs['output']['data']['additional_kwargs']['tool_calls']
    #         output_message['tool_calls'] = tool_calls            
    #     else:
    #         output_message = record.outputs['output']['data']

    #     messages = [to_langsmith_message(m) for m in input_messages] + [to_langsmith_message(output_message, is_output=True)]
    #     conversation = Conversation()
    #     conversation.messages = messages
    #     return conversation

    
    # def to_rag_example(self):        
    #     # key = key=f"""inpute_messages: {[m.content for m in self.messages if m.is_output == False]}"""
    #     return ConversationRagMetadata(            
    #         inputs=json.dumps([m.dict() for m in self.messages[:-1]]),
    #         output=json.dumps(self.messages[-1].dict())
    #     )
    #     # return {
    #     #     "key": f"""inpute_messages: {[m['content'] for m in self.input_messages]}""",
    #     #     "input": [m.to_openai() for m in self.messages[:-1]],
    #     #     "output": self.messages[-1].to_openai()
    #     # }
    
    # @staticmethod
    # def from_rag_example(example: ConversationRagMetadata):
    #     conversation = Conversation()
    #     input_messages = [to_langsmith_message(m) for m in json.loads(example.input)]
    #     output_message = to_langsmith_message(json.loads(example.output), is_output=True)
    #     conversation.messages = input_messages + [output_message]
    #     return conversation




class ConversationRagMessage(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Any]] = None
    
class ConversationRagValue(RagValue):
    messages: List[ConversationRagMessage]


class RagConversationVectorizer:

    def __init__(self) -> None:
        self.dense_embeddings = DenseEmbeddings()
    

    def _stringify_message(self, message: Union[AIMessage, HumanMessage, SystemMessage]):
        if type(message) == AIMessage:
            if message.tool_calls:
                return f"{message.role}: {message.content}\n{message.tool_calls} tool calls"
        return f"{message.role}: {message.content}"

    def _preprocess_key(self, conversation, use_last_message=True):        
        if use_last_message:
            key_messages = conversation.messages[-1:]
        else:
            key_messages = conversation.messages

        str_key_messages = "\n".join([self._stringify_message(m) for m in key_messages])
        return str_key_messages
    
    def _pack_value(self, conversation):
        value_messages = conversation.messages
        rag_messages = []
        for m in value_messages:
            value = ConversationRagMessage(role=m.role, content=m.content)
            if type(m) == AIMessage:
                value.tool_calls = value.tool_calls
            rag_messages.append(value)
        return ConversationRagValue(messages=rag_messages), conversation.id
        

    def _get_document_key_value(self, conversation, include_system=False):
        value_messages = conversation.messages
        if not include_system and type(value_messages[0]) == SystemMessage:
            value_messages = value_messages[1:]        
        key_messages = value_messages[:-1]
        str_key_messages = "\n".join([self._stringify_message(m) for m in key_messages])

        rag_messages = []
        for m in value_messages:
            value = ConversationRagMessage(role=m.role, content=m.content)
            if type(m) == AIMessage:
                value.tool_calls = value.tool_calls
            rag_messages.append(value)
        return str_key_messages, ConversationRagValue(messages=rag_messages), conversation.id
    

    def _get_query_key_value(self, conversation, include_system=False):
        value_messages = conversation.messages
        if not include_system and type(value_messages[0]) == SystemMessage:
            value_messages = value_messages[1:]        
        key_messages = value_messages
        str_key_messages = "\n".join([self._stringify_message(m) for m in key_messages])

        rag_messages = []
        for m in value_messages:
            value = ConversationRagMessage(role=m.role, content=m.content)
            if type(m) == AIMessage:
                value.tool_calls = value.tool_calls
            rag_messages.append(value)
        return str_key_messages, ConversationRagValue(messages=rag_messages), conversation.id


    async def update_document(self, value: Conversation) -> List[RagVector]:
        v, i = self._pack_value(value)
        return RagVector[ConversationRagValue](key=[], value=v, id=i)
        

    # async def embed_documents(self, key_conversations: List[Conversation], value_conversations: List[Conversation]) -> List[RagVector]:        
    async def embed_documents(self, keys: List[Conversation], values: List[Conversation]) -> List[RagVector]:
        embeddings = await asyncio.to_thread(self.dense_embeddings.embed_documents, [self._preprocess_key(kc) for kc in keys])
        value_conversations = [self._pack_value(c) for c in values]
        vectors = [RagVector[ConversationRagValue](key=e, value=v[0], id=v[1]) for e,v in  zip(embeddings, value_conversations)]
        return vectors
    
    
    async def embed_query(self, conversation, use_last_message=True):
        embedding = await asyncio.to_thread(self.dense_embeddings.embed_query, self._preprocess_key(conversation, use_last_message))
        value, conv_id = self._pack_value(conversation)
        return RagVector(key=embedding, value=value)
    
    # async def embed_documents(self, conversations: List[Conversation], include_system=False) -> List[RagVector]:
    #     key_value_conversations = [self._get_document_key_value(c, include_system) for c in conversations]
    #     embeddings = await asyncio.to_thread(self.dense_embeddings.embed_documents, [kv[0] for kv in key_value_conversations])                
    #     vectors = [RagVector[ConversationRagValue](key=e, value=v[1], id=v[2]) for e,v in  zip(embeddings, key_value_conversations)]
    #     return vectors
    

    # async def embed_query(self, conversation, include_system=False):
    #     key, value, conv_id = self._get_query_key_value(conversation, include_system)
    #     embedding = await asyncio.to_thread(self.dense_embeddings.embed_query, key)
    #     return RagVector(key=embedding, value=value)
    

    


class LangsmithConversationDataset:

    def __init__(self, dataset_name) -> None:
        self.name = dataset_name
        self.client = Client()


    def langsmith_to_conversation(self, records, is_agent=False):
        conversation_list = []
        for record in records:
            input_messages = [m['data'] for m in record.inputs['input']]
            if is_agent:
                output_message = input_messages[-1]
                input_messages = input_messages[:-1]
                tool_calls = record.outputs['output']['data']['additional_kwargs']['tool_calls']
                output_message['tool_calls'] = tool_calls            
            else:
                output_message = record.outputs['output']['data']

            messages = [to_langsmith_message(m) for m in input_messages] + [to_langsmith_message(output_message, is_output=True)]
            conversation = Conversation(conv_id=str(record.id))
            conversation.messages = messages
            conversation_list.append(conversation)
        return conversation_list
    
    def get_records(self, is_agent=False):
        datasets = self.client.list_datasets(dataset_name=self.name)
        dataset = list(datasets)[0]
        examples = self.client.list_examples(dataset_id=dataset.id)
        return self.langsmith_to_conversation(list(examples), is_agent=is_agent)
    



class ConversationRag:

    def __init__(self, name, is_state_value=False) -> None:
        self.name = name
        self.rag_vectorizer = RagConversationVectorizer()
        self.rag_space = RagVectorSpace(name, self.rag_vectorizer, ConversationRagValue)


    async def add_conversation_list(self, conversations: List[Conversation]):
        return await self.rag_space.add_many(conversations)
    
    async def add_examples(self, examples: List[Dict[str, Conversation]]):
        return await self.rag_space.add_many(examples)
    
    async def update_example(self, example_id: str, conversation: Conversation):
        return await self.rag_space.update(example_id, conversation)

    async def add_conversation(self, conversation: Conversation):
        return await self.rag_space.add_many([conversation])
    
    def add_example_view(self, idx, content):
        return f"""
EXAMPLE {idx}:
{content}
"""
    def unpack_examples(self, ex):
        messages = []
        for msg in ex.value.messages:
            if msg.role == "assistant":
                messages.append(AIMessage(content=msg.content, tool_calls=msg.tool_calls))
            elif msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
        return Conversation(messages=messages)


    async def similarity(self, conversation: Conversation, top_k=3):
        # query = await self.rag_vectorizer.embed_query(conversation)
        examples = await self.rag_space.similarity(conversation, top_k=top_k)
        rag_conversations = []
        for idx, ex in enumerate(examples):
            messages = []
            for msg in ex.value.messages:
                if msg.role == "assistant":
                    messages.append(AIMessage(content=self.add_example_view(idx + 1, msg.content), tool_calls=msg.tool_calls))
                elif msg.role == "user":
                    messages.append(HumanMessage(content=self.add_example_view(idx + 1, msg.content)))
                elif msg.role == "system":
                    messages.append(SystemMessage(content=self.add_example_view(idx + 1, msg.content)))
            rag_conversations.append(Conversation(messages=messages))
                    
        return rag_conversations
    
    
    async def get_many(self, top_k=100):
        res = await self.rag_space.get_many(top_k=top_k)
        return res
    

    async def delete_all(self):
        return await self.rag_space.delete_all()
    

    async def delete(self, example_id):
        return await self.rag_space.delete(example_id)
