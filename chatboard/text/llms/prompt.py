
import asyncio
from enum import Enum
import inspect
import os
import pathlib
import random
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar, Generic, Union


from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langsmith.run_trees import RunTree
from jinja2 import Environment, FileSystemLoader, PackageLoader, meta
from git import GitCommandError, Repo

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain.callbacks.manager import AsyncCallbackManagerForChainGroup
from langchain.callbacks.manager import atrace_as_chain_group, trace_as_chain_group
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from components.etl.completion_parsing import auto_split_completion, parse_completion
from components.etl.llm_cache import LlmStreamCache
from components.etl.rag_legacy import RagManager
from components.vectors.pinecone_text_vector_store import PineconeTextVectorStore
from config import PROMPT_REPO_HOME
import tiktoken
import yaml
import re




# model_name = 'text-embedding-ada-002'

# embed = OpenAIEmbeddings(
#     document_model_name=model_name,
#     query_model_name=model_name,
#     openai_api_key=OPENAI_API_KEY
# )



# llm = HermesLLM()


class PromptException(Exception):

    def __init__(self, message, completion) -> None:
        self.message = message
        self.completion = completion
        super().__init__(f"{self.message} | {completion}")



class PromptTemplateManager:

    def __init__(self, directory, filename=None, config=None) -> None:
        self.config = config
        self.env = Environment(loader=FileSystemLoader(directory))
        if filename is not None and (filename.endswith('.jinja2') or filename.endswith('.jinja')):
            self.template_names = [filename]
        else:    
            self.template_names = [ t for t in self.env.list_templates() if re.match(r'.*\.jinja', t) and t not in config['ignore'] ]
        self.template_file = filename
        self.directory = directory
        # if directory.name.endswith('.jinja2') or directory.name.endswith('.jinja'):
        #     self.template_names = [directory.name]
        #     self.directory = directory.parent
        #     self.env = Environment(loader=FileSystemLoader(directory.parent))
        # else:
        #     self.env = Environment(loader=FileSystemLoader(directory))
        #     self.template_names = [ t for t in self.env.list_templates() if re.match(r'.*\.jinja', t) ]
        #     self.template_file = template_file
        #     self.directory = directory


    def get_random_template_file(self, template_type=None, filename=None):
        if not self.template_names:
            raise Exception("No templates found in prompt directory")
        
        if template_type == "system":
            optional_templetes = [t for t in self.template_names if t.startswith('system')]
        elif template_type == "user":
            optional_templetes = [t for t in self.template_names if t.startswith('user')]
        else:
            optional_templetes = self.template_names
        if filename:
            if filename in optional_templetes:
                return filename
            else:
                raise Exception(f"filename {filename} not found in prompt directory")
        template_name = random.choice(optional_templetes)
        return template_name
        

    def get_template_source(self, template_name=None):
        # if not template_name: 
        #     if self.template_file:
        #         template_name = self.template_file
        #     else:            
        #         template_name = self.template_names[0]
        #         self.template_file = template_name
        template_source = self.env.loader.get_source(self.env, template_name)[0]
        return template_source
        

    def get_template(self, template_name=None):
        template_source = self.get_template_source(template_name)
        return PromptTemplate.from_template(template_source, template_format="jinja2")


    def get_template_variables(self, template_name=None):
        template_source = self.get_template_source(template_name)
        parsed_content = self.env.parse(template_source)
        # return list(meta.find_undeclared_variables(parsed_content))
        return list(meta.find_undeclared_variables(parsed_content))
    



default_prompt_config = {
    "model": 'gpt-3.5-turbo-1106',
    "temperature": 0,
    "max_tokens": None,
    "num_examples": 3,
    "stop_sequences": None,
    "ignore": [],
}



chat_models = [
    'gpt-3.5-turbo',
    'gpt-4-1106-preview',
    'gpt-3.5-turbo-1106',
    "gpt-4-0125-preview",
]

def encode_logits(string: str, bias_value: int, encoding_model: str) -> int:
    """Returns the number of tokens in a text string."""
    
    return {en: bias_value for en in encoding_model.encode(string)}



def encode_logits_dict(logits, encoding_model):
    encoded_logits = {}
    for key, value in logits.items():
        item_logits = encode_logits(key, value, encoding_model)
        encoded_logits.update(item_logits)
    return encoded_logits


class PromptManager:

    # def __init__(self, promptpath=None, filename=None, caller_dir=None):
    #     self.promptpath = None
    #     self.filename = None
    #     if promptpath:
    #         self.promptpath = pathlib.Path(promptpath)
    #     if filename:
    #         self.filename = filename
    #         if not filename.endswith('.jinja2') or filename.endswith('.jinja'):
    #             raise Exception("filename must end with .jinja2 or .jinja")
    #         self.prompt_path = caller_dir / filename
            
    #     self.configpath = None
    #     prompt_config = default_prompt_config.copy()
    #     if self.promptpath:
    #         configpath =  PROMPT_REPO_HOME / self.promptpath / 'config.yaml'
    #         if configpath.exists():                
    #             prompt_config.update(yaml.safe_load(open(configpath)))
    #         self.configpath = configpath

    #     self.prompt_config = prompt_config
    #     self.template_manager = PromptTemplateManager(PROMPT_REPO_HOME / self.promptpath)
    #     if not self.filename:
    #         self.prompt_repo = Repo(PROMPT_REPO_HOME)
    #     else:
    #         self.prompt_repo = None

    def __init__(self, promptpath: pathlib.Path =None):
        self.prompt_repo = None
        self.configpath = None
        encoding_name = "cl100k_base"
        self.encoding_model = tiktoken.get_encoding(encoding_name)
        
        prompt_config = default_prompt_config.copy()
        if promptpath.name.endswith('.jinja2') or promptpath.name.endswith('.jinja'):
            self.promptpath = promptpath.parent
            self.filename = promptpath.name
        else:
            self.promptpath = promptpath
            self.filename = None
        configpath = self.promptpath / 'config.yaml'
        if configpath.exists():                
            prompt_config.update(yaml.safe_load(open(configpath)))
        self.configpath = configpath
        self.prompt_config = prompt_config
        self.template_manager = PromptTemplateManager(
            self.promptpath, 
            filename=self.filename,
            config=prompt_config
        )
        try:
            self.prompt_repo = Repo(PROMPT_REPO_HOME) if not self.filename else None
        except:
            self.prompt_repo = None
            


    def get_template(self, is_chat_template=False, template_type=None, filename=None):        
        if is_chat_template:
            if template_type == "system":
                template_file = self.template_manager.get_random_template_file(template_type=template_type, filename=filename)
                input_variables = self.template_manager.get_template_variables(template_file)
                prompt_template = SystemMessagePromptTemplate.from_template_file(
                    self.promptpath / template_file,
                    template_format="jinja2", 
                    input_variables=input_variables
                )
            elif template_type == "user":
                template_file = self.template_manager.get_random_template_file(template_type=template_type, filename=filename)
                input_variables = self.template_manager.get_template_variables(template_file)
                prompt_template = HumanMessagePromptTemplate.from_template_file(
                    self.promptpath / template_file,
                    template_format="jinja2", 
                    input_variables=input_variables
                )
        else:
            template_file = self.template_manager.get_random_template_file(template_type=template_type, filename=filename)
            input_variables = self.template_manager.get_template_variables(template_file)
            prompt_template = PromptTemplate.from_file(
                self.promptpath / template_file,
                template_format="jinja2", 
                input_variables=input_variables
            )
        
        commit_hex = None
        commit_message = None
        if self.prompt_repo:
            commits = self.prompt_repo.iter_commits(paths=self.promptpath / template_file)
            commits = list(commits)
            if len(commits) > 0:
                commit_hex = commits[0].hexsha[:7]
                commit_message = commits[0].message
            
        metadata = {
            'prompt': template_file.replace('.jinja2', '').replace('.jinja', ''),
            'commit': commit_hex,
            'commit_message': commit_message,
        }
        return prompt_template, metadata

    @property
    def model(self):
        return self.prompt_config['model']
    
    @property
    def temperature(self):
        return self.prompt_config['temperature']
    
    @property
    def max_tokens(self):
        return self.prompt_config['max_tokens']
    
    @property
    def num_examples(self):
        return self.prompt_config['num_examples']
    
    @property
    def stop_sequences(self):
        return self.prompt_config['stop_sequences']
    
    def get_llm(
            self, 
            model: str=None, 
            llm_params=None, 
            stop_sequences: List[str]=None, 
            temperature: float=None, 
            max_tokens: int=None, 
            streaming: bool=False,
            logit_bias: Dict[str, float]=None,
            top_p: float=None,
            presence_penalty: float=None,
            frequency_penalty: float=None,
            suffix: str=None,
        ):
        if model is None:
            model = self.model
        model_kwargs={}
        if llm_params:
            model_kwargs.update(llm_params)
        if stop_sequences:
            model_kwargs['stop'] = stop_sequences
        if logit_bias:            
            model_kwargs['logit_bias'] = encode_logits_dict(logit_bias, self.encoding_model)
        if top_p:
            if top_p > 1.0 or top_p < 0.0:
                raise ValueError("top_p must be between 0.0 and 1.0")
            model_kwargs['top_p'] = top_p

        if presence_penalty:
            if presence_penalty > 2.0 or presence_penalty < -2.0:
                raise ValueError("presence_penalty must be between -2.0 and 2.0")
            model_kwargs['presence_penalty'] = presence_penalty
        if frequency_penalty:
            if frequency_penalty > 2.0 or frequency_penalty < -2.0:
                raise ValueError("frequency_penalty must be between -2.0 and 2.0")
            model_kwargs['frequency_penalty'] = frequency_penalty
        if suffix:
            model_kwargs['suffix'] = suffix
        if model in chat_models:
            return ChatOpenAI(
                model=model,
                temperature=temperature or 0, 
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                streaming=streaming,
            )
        else:    
            return OpenAI(
                model_name=model,
                temperature=temperature, 
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
                streaming=streaming,
            )



def to_dict(pydantic_model):
    d = {}
    for field_name, field_info in pydantic_model.__fields__.items():
        d[field_name] = None
    return d        


class PromptChunkTypes(Enum):
    PROMPT_START = "prompt_start"
    PROMPT_UPDATE = "prompt_update"
    PROMPT_FINISH = "prompt_finish"
    PROMPT_ERROR = "prompt_error"




def sanitize_content(content):
    content = content.strip().strip('"').strip("'").strip('\n')
    return content



class PromptErrorTypes(Enum):
    PYDANTIC_ERROR = "pydantic_error"

class PromptError(BaseModel):
    error: str
    error_type: PromptErrorTypes


class PromptResponse(BaseModel):
    value: Any
    run_id: str
    completion: str
    messages: List[BaseMessage]
    costs: Optional[Any] = None
    model: Optional[str] = None
    error: Optional[PromptError] = None

    def __repr__(self) -> str:
        return f"{self.value!r}"
    
    def __str__(self) -> str:
        return f"{self.value!r}"
    
    def to_dict(self):
        if hasattr(self.value, 'to_dict') and callable(getattr(self.value, 'to_dict')):
            value = self.value.to_dict()
        else:
            value = self.value

        return {
            "value": value,
            "run_id": self.run_id,
            "completion": self.completion,
            # "messages": [m.to_dict() for m in self.messages],
            # "costs": self.costs,
            # "model": self.model
        }


class PromptChunk(BaseModel):
    msg_type: PromptChunkTypes
    func: Optional[str] = None
    content: Optional[Union[str, BaseModel]]
    prompt_response: Optional[PromptResponse] = None
    field: Optional[str]

    def to_dict(self):
        return {
            "msg_type": self.msg_type.value,
            "func": self.func,
            "content": self.content,
            "prompt_response": self.prompt_response.to_dict() if self.prompt_response is not None else None,
            "field": self.field
        }



# class PromptResponse:
#     def __init__(
#             self, 
#             value, 
#             run_id, 
#             completion, 
#             messages, 
#             model,
#             costs=None
#         ):
#         self.value = value      
#         self.run_id = run_id
#         self.completion = completion
#         self.messages = messages
#         self.costs = costs
#         self.model = model

#     def __repr__(self) -> str:
#         return self.value.__repr__()
    
#     def __str__(self) -> str:
#         return self.value.__repr__()

ValueType = TypeVar('ValueType')

# class PromptResponse(GenericModel, Generic[ValueType]):
#     value: ValueType
#     run_id: str
#     completion: str
#     messages: List[BaseMessage]
#     costs: Optional[Any] = None
#     model: Optional[str] = None

#     def __repr__(self) -> str:
#         return f"{self.value!r}"
    
#     def __str__(self) -> str:
#         return f"{self.value!r}"




class StreamPrompt:
    pass


class InstructPrompt:
    pass




class prompt:
    """

    frequency, presence and logit bias equation:
        mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
    where:
        mu[j] is the logits of the j-th token
        c[j] is how often that token was sampled prior to the current position
        float(c[j] > 0) is 1 if c[j] > 0 and 0 otherwise
        alpha_frequency is the frequency penalty coefficient
        alpha_presence is the presence penalty coefficient

    """
    

    def __init__(
            self,
            # promptpath,
            promptpath=None,
            _func=None, 
            params={}, 
            filename=None,        
            # model='gpt-3.5-turbo-1106', 
            model=None,
            temperature=None, 
            max_tokens=None, 
            num_examples=3, 
            run_name=None,
            rag_index=None,
            rag_query_key=None,
            rag_var="examples",
            rag_fields=None,
            stop_sequences=None,
            stream=False,
            pydantic_model=None,
            logit_bias=None,            
            top_p=None,
            presence_penalty=None,
            frequency_penalty=None,
            suffix=None,

        ):
        caller_frame = inspect.stack()[1]
        caller_filepath = pathlib.Path(caller_frame.filename)
        self.caller_dir = caller_filepath.parent
        if filename:
            filename = pathlib.Path(filename)
            self.promptpath = filename if filename.is_absolute() else self.caller_dir / filename
            # if filename.is_absolute():
            #     self.promptpath = filename
            # else:
            #     self.promptpath = self.caller_dir / filename
        else:
            self.promptpath = pathlib.Path(PROMPT_REPO_HOME) / promptpath
        self._func = _func
        self.params = params
        self.filename = filename
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_examples = num_examples
        self.run_name = run_name
        self.rag_index = rag_index
        self.rag_query_key = rag_query_key or 'prompt'
        self.rag_var = rag_var
        self.rag_fields = rag_fields
        self.stop_sequences = stop_sequences
        self.add_context = False
        self.stream = stream
        self.pydantic_model = pydantic_model
        self.logit_bias = logit_bias
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.suffix = suffix
        
        if self.rag_index:
            # if not self.rag_query_key:                
                # raise Exception("rag_query_key is required when using rag_index")
            # self.vector_store = PineconeTextVectorStore('rag', self.rag_index)
            self.rag_manager = RagManager(self.rag_index)
        # self.prompt_manager = PromptManager(promptpath)


    def build_llm_chain(
            self, 
            func, 
            llm=None, 
            run_manager=None, 
            promptpath=None, 
            llm_params=None,
            filename=None,
            temperature=None,
            max_tokens=None,
            model=None,
            top_p=None,
            presence_penalty=None,
            frequency_penalty=None,
            suffix=None,
        ):

        # class CustomOutputParser(BaseOutputParser):
        #         """Parse the output of an LLM call to a comma-separated list."""
        #         def parse(self, text: str):
        #             if inspect.iscoroutinefunction(func):
        #                 return asyncio.run(func(text=text))
        #             return func(text=text)         

        
        if promptpath is not None:
            if type(promptpath) == str:
                promptpath = pathlib.Path(promptpath)
            promptpath = pathlib.Path(PROMPT_REPO_HOME) / promptpath
        else:
            promptpath = self.promptpath

        if filename is not None:
            if type(filename) == str:
                filename = pathlib.Path(filename)
            promptpath = promptpath / filename
                
        prompt_manager = PromptManager(promptpath)

        if not llm:
            llm = prompt_manager.get_llm(
                model=model or self.model, 
                llm_params=llm_params, 
                stop_sequences=self.stop_sequences, 
                temperature=temperature, 
                max_tokens=max_tokens,
                streaming=self.stream,
                logit_bias=self.logit_bias,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                suffix=suffix,
            )

        prompt_template, prompt_metadata = prompt_manager.get_template()            

        chain = (prompt_template | llm)
        config = {
                "metadata": prompt_metadata,
            }        
        if run_manager:
            config['callbacks' ] = run_manager            

        return chain, config
    
    async def build_chat_llm_chain(
            self, 
            func, 
            llm=None, 
            run_manager=None, 
            promptpath=None, 
            llm_params=None,
            system_filename=None,
            user_filename=None,
            temperature=None,
            max_tokens=None,
            model=None,
            logit_bias=None,
            top_p=None,
            presence_penalty=None,
            frequency_penalty=None,
            suffix=None,
            **kwargs
        )-> Tuple[ChatOpenAI, List[BaseMessage], Any]:

        # class CustomOutputParser(BaseOutputParser):
        #         """Parse the output of an LLM call to a comma-separated list."""
        #         def parse(self, text: str):
        #             if inspect.iscoroutinefunction(func):
        #                 return asyncio.run(func(text=text))
        #             return func(text=text)         

        
        if promptpath is not None:
            if type(promptpath) == str:
                promptpath = pathlib.Path(promptpath)
            promptpath = pathlib.Path(PROMPT_REPO_HOME) / promptpath
        else:
            promptpath = self.promptpath

        # if filename is not None:
        #     if type(filename) == str:
        #         filename = pathlib.Path(filename)
        #     promptpath = promptpath / filename
                
        prompt_manager = PromptManager(promptpath)

        if not llm:
            llm = prompt_manager.get_llm(
                model=model or self.model, 
                llm_params=llm_params, 
                stop_sequences=self.stop_sequences, 
                temperature=temperature, 
                max_tokens=max_tokens,
                streaming=self.stream,
                logit_bias=logit_bias or self.logit_bias,
                top_p=top_p or self.top_p,
                presence_penalty=presence_penalty or self.presence_penalty,
                frequency_penalty=frequency_penalty or self.frequency_penalty,
                suffix=suffix or self.suffix,
            )
        examples = []
        if self.rag_index:
            examples = await self.retrieve_rag_async(kwargs['kwargs'])
        
        prompt_system_template, prompt_metadata = prompt_manager.get_template(is_chat_template=True, template_type="system", filename=system_filename)            
        # prompt_value = prompt_template.invoke(kwargs['kwargs'])
        system_message = prompt_system_template.format(**kwargs['kwargs'])

        messages = [
            system_message    
        ]
        for i, example in enumerate(examples):
            messages.append(HumanMessage(
                content=f"Example {i + 1}:\n {example.key}",
                example=True
            ))
            messages.append(AIMessage(
                content=f"Example {i + 1}:\n{example.text}",
                example=True
            ))
            if example.feedback is not None:
                messages.append(HumanMessage(
                    content=f"Example {i + 1}:\n{example.feedback}",
                    example=True
                ))

        if kwargs["kwargs"].get("prompt", None):
            messages.append(
                HumanMessage(
                    content=kwargs["kwargs"]["prompt"],
                )
            )
        else:
            prompt_user_template, prompt_metadata = prompt_manager.get_template(is_chat_template=True, template_type="user", filename=user_filename)        
            user_message = prompt_user_template.format(**kwargs['kwargs'])
            messages.append(user_message)

        config = {
                "metadata": prompt_metadata,
            }        
        if run_manager:
            config["callbacks"] = run_manager

        return llm, messages, config
    
    
    def retrieve_rag(self, kwargs):
        kwargs_copy = kwargs.copy()
        examples_res = self.rag_manager.search(
            query=kwargs_copy[self.rag_query_key],
            top_k=self.num_examples, 
            namespace=self.rag_index
        )
        # examples_res = self.vector_store.text_similarity_search(
        #     text=kwargs_copy[self.rag_query_key],
        #     top_k=self.num_examples, 
        #     namespace=self.rag_index
        # )
        # examples_res = []
        
        # examples = [r.metadata['text'] for r in examples_res]
        examples = [r.metadata.text for r in examples_res]
        kwargs_copy[self.rag_var] = examples
        return kwargs_copy
    

    async def retrieve_rag_async(self, kwargs):
        # kwargs_copy = kwargs.copy()        
        examples_res = await self.rag_manager.search(
            query=kwargs[self.rag_query_key],
            top_k=self.num_examples, 
            namespace=self.rag_index
        )        
        examples = [r.metadata for r in examples_res]
        # kwargs_copy[self.rag_var] = examples
        return examples
    
    
    def __call__(self, func): 

        signature = inspect.signature(func)
        parameters = signature.parameters
        argument_names = [param for param in parameters if parameters[param].default == inspect.Parameter.empty]
        if 'context' in argument_names:
            self.add_context = True

        async def stream_wrapper(
                *args, 
                llm=None, 
                llm_params=None, 
                run_manager=None, 
                promptpath=None, 
                filename=None, 
                temperature=None, 
                max_tokens=None,
                model=None,
                logit_bias=None,
                top_p=None,
                presence_penalty=None,
                frequency_penalty=None,
                suffix=None,
                **kwargs
            ):
            # if self.rag_index:
                # kwargs = await self.retrieve_rag_async(kwargs)
            
            async with atrace_as_chain_group(
                    func.__name__,
                    callback_manager=run_manager,
                    inputs=kwargs,
                ) as group_manager:

                chat, messages, config = await self.build_chat_llm_chain(
                        func, 
                        llm=llm, 
                        # run_manager=run_manager, 
                        run_manager=group_manager,
                        promptpath=promptpath,
                        filename=filename,
                        llm_params=llm_params,
                        temperature=temperature or self.temperature, 
                        max_tokens=max_tokens or self.max_tokens,
                        model=model,
                        logit_bias=logit_bias or self.logit_bias,
                        top_p=top_p or self.top_p,
                        presence_penalty=presence_penalty or self.presence_penalty,
                        frequency_penalty=frequency_penalty or self.frequency_penalty,
                        suffix=suffix or self.suffix,
                        kwargs=kwargs
                )
                with get_openai_callback() as cb:
                    if self.stream:
                        llm_stream_cache = LlmStreamCache(run_name=func.__name__, run_id=group_manager.parent_run_id)
                        if self.pydantic_model:
                            output = to_dict(self.pydantic_model)
                            curr_field = None
                            curr_content = ""
                            full_content = ""
                            message = PromptChunk(
                                msg_type=PromptChunkTypes.PROMPT_START,
                                func=func.__name__,
                                content=None,
                                field=None
                            )
                            yield message
                            async for cnk in chat.astream(messages, config=config):
                                content_chunk = cnk.content
                                llm_stream_cache.add(content_chunk)
                                # print("-->", content_chunk)
                                full_content+=content_chunk
                                if content_chunk == None:
                                    break
                                yield PromptChunk(
                                    msg_type=PromptChunkTypes.PROMPT_UPDATE,
                                    func=func.__name__,
                                    content=content_chunk,
                                    field=curr_field
                                )
                                output, curr_field, curr_content = auto_split_completion(curr_content, content_chunk, output, curr_field, self.pydantic_model)
                                
                                
                            else:
                                output[curr_field] = sanitize_content(curr_content)
                            # try:
                            if self.pydantic_model:
                                output = self.pydantic_model(**output)

                            if self.add_context:
                                parsed_output = await func(output, context={'kwargs': kwargs})
                            else:
                                parsed_output = await func(output)                        
                                
                            response = PromptResponse(
                                value=parsed_output,
                                run_id=str(group_manager.parent_run_id),
                                completion=full_content,
                                messages=messages,
                                model=model or self.model,
                                # costs=costs
                            )
                            
                            yield PromptChunk(
                                msg_type=PromptChunkTypes.PROMPT_FINISH,
                                func=func.__name__,
                                # content=response,
                                prompt_response=response,
                                field=None
                            )
                            llm_stream_cache.save(full_content)
                            # except Exception as e:
                                # print("error:", e)
                                # print("#", cnk.content)
                                # yield func_iterator.send(cnk.content)
                                # output += cnk.content
                            # parsed_output = await func(output)
                            # yield parsed_output
                            return
                        else:
                            full_content = ""
                            yield PromptChunk(
                                msg_type=PromptChunkTypes.PROMPT_START,
                                func=func.__name__,
                                content=None,
                                field=None
                            )
                            async for cnk in chat.astream(messages, config=config):
                                content_chunk = cnk.content
                                # print("-->", content_chunk)
                                full_content+=content_chunk
                                if content_chunk == None:
                                    break
                                yield PromptChunk(
                                    msg_type=PromptChunkTypes.PROMPT_UPDATE,
                                    func=func.__name__,
                                    content=content_chunk,
                                )
                            yield PromptChunk(
                                msg_type=PromptChunkTypes.PROMPT_FINISH,
                                func=func.__name__,
                                # content=response,
                                prompt_response=PromptResponse(
                                    value=full_content,
                                    run_id=str(group_manager.parent_run_id),
                                    completion=full_content,
                                    messages=messages,
                                    model=model or self.model,
                                    # costs=costs
                                ),
                                field=None
                            )
        if self.stream:
            return stream_wrapper


        async def async_wrapper(
                *args, llm=None, 
                llm_params=None, 
                run_manager: AsyncCallbackManagerForChainGroup=None, 
                promptpath=None, 
                system_filename=None, 
                user_filename=None,
                temperature=None, 
                max_tokens=None,
                completion=None,
                model=None,
                logit_bias=None,
                top_p=None,
                presence_penalty=None,
                frequency_penalty=None,
                suffix=None,
                **kwargs
            ):
            if completion:
                if self.add_context:
                    return await func(completion, context={'kwargs': kwargs})
                else:
                    return await func(completion)

            async with atrace_as_chain_group(
                func.__name__,
                callback_manager=run_manager,
                inputs=kwargs,
            ) as group_manager:
                
                chat, messages, config = await self.build_chat_llm_chain(
                    func, 
                    llm=llm, 
                    # run_manager=run_manager, 
                    run_manager=group_manager,
                    promptpath=promptpath,
                    system_filename=system_filename,
                    user_filename=user_filename,
                    llm_params=llm_params,
                    temperature=temperature or self.temperature, 
                    max_tokens=max_tokens or self.max_tokens,
                    model=model,
                    logit_bias=logit_bias,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    suffix=suffix,
                    kwargs=kwargs
                )

                costs = None
                with get_openai_callback() as cb:
                        # output = await chain.ainvoke(kwargs, config=config)                
                        output = await chat.ainvoke(messages, config=config)
                        if self.pydantic_model:
                            try:
                                content = parse_completion(output.content, self.pydantic_model)
                            except Exception as e:
                                return PromptResponse(
                                    value=output.content,
                                    run_id=str(group_manager.parent_run_id),
                                    completion=output.content,
                                    error=PromptError(error=str(e), error_type=PromptErrorTypes.PYDANTIC_ERROR),
                                )
                                # raise PromptException(f"failed parsing pydantic model{self.pydantic_model}", output.content)
                        else:
                            content = output.content
                        kwargs['run_id'] = str(group_manager.parent_run_id)
                        if self.add_context:
                            parsed_output = await func(content, context={'kwargs': kwargs})
                        else:
                            parsed_output = await func(content)
                        costs={
                            "cost": cb.total_cost,
                            "tokens": cb.total_tokens,
                            "prompt_tokens": cb.prompt_tokens,
                            "completion_tokens": cb.completion_tokens,
                        }
                print('cost:', cb.total_cost, 'tokens:', cb.total_tokens, 'prompt:',cb.prompt_tokens, 'completion:', cb.completion_tokens)
                # group_manager.on_chain_end({"output": parsed_output})
                return PromptResponse(
                    value=parsed_output,
                    run_id=str(group_manager.parent_run_id),
                    completion=output.content,
                    messages=messages,
                    model=model or self.model,
                    costs=costs
                )
        # async def async_wrapper(
        #         *args, llm=None, 
        #         llm_params=None, 
        #         run_manager: AsyncCallbackManagerForChainGroup=None, 
        #         promptpath=None, 
        #         filename=None, 
        #         temperature=None, 
        #         max_tokens=None,
        #         completion=None,
        #         model=None,
        #         **kwargs
        #     ):

        #     if completion:
        #         return await func(completion, context={'kwargs': kwargs})

        #     if self.rag_index:
        #         kwargs = await self.retrieve_rag_async(kwargs)

        #     async with atrace_as_chain_group(
        #         func.__name__,
        #         callback_manager=run_manager,
        #         inputs=kwargs,
        #     ) as group_manager:

        #         chain, config = self.build_llm_chain(
        #             func, 
        #             llm=llm, 
        #             # run_manager=run_manager, 
        #             run_manager=group_manager,
        #             promptpath=promptpath,
        #             filename=filename,
        #             llm_params=llm_params,
        #             temperature=temperature or self.temperature, 
        #             max_tokens=max_tokens or self.max_tokens,
        #             model=model,
        #         )

        #         costs = None
        #         with get_openai_callback() as cb:
        #                 output = await chain.ainvoke(kwargs, config=config)                
        #                 kwargs['run_id'] = str(group_manager.parent_run_id)
        #                 if self.add_context:
        #                     parsed_output = await func(output.content, context={'kwargs': kwargs})
        #                 else:
        #                     parsed_output = await func(output.content)                            
        #                 costs={
        #                     "cost": cb.total_cost,
        #                     "tokens": cb.total_tokens,
        #                     "prompt_tokens": cb.prompt_tokens,
        #                     "completion_tokens": cb.completion_tokens,
        #                 }
        #         print('cost:', cb.total_cost, 'tokens:', cb.total_tokens, 'prompt:',cb.prompt_tokens, 'completion:', cb.completion_tokens)
        #         return PromptResponse(
        #             value=parsed_output,
        #             run_id=group_manager.parent_run_id,
        #             completion=output.content,
        #             costs=costs
        #         )

            
            
        def wrapper(
                *args, 
                llm=None, 
                llm_params=None, 
                run_manager=None, 
                promptpath=None, 
                filename=None, 
                temperature=None, 
                max_tokens=None,
                completion=None,
                model=None,
                **kwargs
            ):

            if completion:
                return func(completion, context={'kwargs': kwargs})

            if self.rag_index:
                kwargs = self.retrieve_rag(kwargs)

            chain, config = self.build_llm_chain(
                func, 
                llm=llm, 
                run_manager=run_manager, 
                promptpath=promptpath,
                filename=filename,
                llm_params=llm_params,
                temperature=temperature or self.temperature,  
                max_tokens=max_tokens or self.max_tokens,
                model=model,
            )
            with get_openai_callback() as cb:
                output = chain.invoke(kwargs, config=config)
                print('cost:', cb.total_cost, 'tokens:', cb.total_tokens, 'prompt:',cb.prompt_tokens, 'completion:', cb.completion_tokens)
                print("$$$$", argument_names)
                # if self.add_context:
                #     parsed_output = func(output['text'], context={'kwargs': kwargs})
                if self.add_context:
                    parsed_output = func(output.content, context={'kwargs': kwargs})
                else:
                    parsed_output = func(output.content)
                return parsed_output
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    

            
    # def async_run(self,)
    
    # if _func is None:
    #     return decorator_prompt
    # else:
    #     return decorator_prompt(_func)

# def prompt(
#         # promptpath,
#         promptpath=None,
#         _func=None, 
#         params={}, 
#         filename=None,        
#         # model='gpt-3.5-turbo-1106', 
#         model=None,
#         temperature=None, 
#         max_tokens=None, 
#         num_examples=None, 
#         run_name=None,
#         rag_index=None,
#         rag_query_key=None,
#         rag_var="examples",
#         rag_fields=None,
#         stop_sequences=None,
#     ):
#     caller_frame = inspect.stack()[1]
#     caller_filepath = pathlib.Path(caller_frame.filename)
#     caller_dir = caller_filepath.parent

    # def decorator_prompt(func): 
    #     def wrapper_prompt(*args, llm=None, run_manager=None, **kwargs):
    #         class CustomOutputParser(BaseOutputParser):
    #             """Parse the output of an LLM call to a comma-separated list."""

    #             def parse(self, text: str):
    #                 return func(text=text) 
                
    #         # template_manager = PromptTemplateManager(str(caller_dir / filename))
            
    #         prompt_manager = PromptManager(promptpath)

    #         if not llm:
    #             llm = prompt_manager.get_llm(model=model, stop_sequences=stop_sequences)

    #         if rag_index:
    #             if not rag_query_key:
    #                 raise Exception("rag_query_key is required when using rag_index")
                
    #             vector_store = PineconeTextVectorStore('rag', rag_index)
    #             # loop = asyncio.get_event_loop()
    #             # examples_res = loop.run_until_complete(
    #             #     vector_store.atext_similarity_search(
    #             #     text=kwargs[rag_query_key],
    #             #     top_k=num_examples, 
    #             #     namespace=rag_index
    #             # ))
                
    #             examples_res = vector_store.text_similarity_search(
    #                 text=kwargs[rag_query_key],
    #                 top_k=num_examples, 
    #                 namespace=rag_index
    #             )
    #             examples_res = []
                
    #             examples = [r['metadata'] for r in examples_res]
    #             kwargs[rag_var] = examples

            
    #         # input_variables = template_manager.get_template_variables()
    #         prompt_template, prompt_metadata = prompt_manager.get_template()            

    #         chain = LLMChain(
    #             llm=llm,
    #             prompt=prompt_template,
    #             output_parser=CustomOutputParser()
    #         ).with_config({"run_name": run_name or func.__name__})
            

    #         config = {
    #             "metadata": prompt_metadata,
    #         }
    #         if run_manager:
    #             config['callbacks' ] = run_manager                


    #         with get_openai_callback() as cb:
    #             output = chain.invoke(kwargs, config=config)
    #             print('cost:', cb.total_cost, 'tokens:', cb.total_tokens, 'prompt:',cb.prompt_tokens, 'completion:', cb.completion_tokens)
    #             return output['text']
                
    #     return wrapper_prompt
    
#     if _func is None:
#         return decorator_prompt
#     else:
#         return decorator_prompt(_func)




