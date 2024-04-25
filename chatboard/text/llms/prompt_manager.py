

import pathlib
import random


from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from jinja2 import Environment, FileSystemLoader, PackageLoader, meta
from git import GitCommandError, Repo


from .system_conversation import AIMessage, HumanMessage, SystemMessage
import tiktoken
import yaml
import re
import os


PROMPT_REPO_HOME = os.getenv('PROMPT_REPO_HOME', 'prompt_repo')



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





class PromptManager:


    def __init__(self, promptpath: pathlib.Path =None, logit_bias=None):
        self.prompt_repo = None
        self.configpath = None                
        
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
            'input_variables': input_variables,
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
    
    # def get_llm(
    #         self, 
    #         model: str=None, 
    #         llm_params=None, 
    #         stop_sequences: List[str]=None, 
    #         temperature: float=None, 
    #         max_tokens: int=None, 
    #         streaming: bool=False,
    #         logit_bias: Dict[str, float]=None,
    #         top_p: float=None,
    #         presence_penalty: float=None,
    #         frequency_penalty: float=None,
    #         suffix: str=None,
    #     ):
    #     if model is None:
    #         model = self.model
    #     model_kwargs={}
    #     if llm_params:
    #         model_kwargs.update(llm_params)
    #     if stop_sequences:
    #         model_kwargs['stop'] = stop_sequences
    #     if logit_bias:            
    #         model_kwargs['logit_bias'] = encode_logits_dict(logit_bias, self.encoding_model)
    #     if top_p:
    #         if top_p > 1.0 or top_p < 0.0:
    #             raise ValueError("top_p must be between 0.0 and 1.0")
    #         model_kwargs['top_p'] = top_p

    #     if presence_penalty:
    #         if presence_penalty > 2.0 or presence_penalty < -2.0:
    #             raise ValueError("presence_penalty must be between -2.0 and 2.0")
    #         model_kwargs['presence_penalty'] = presence_penalty
    #     if frequency_penalty:
    #         if frequency_penalty > 2.0 or frequency_penalty < -2.0:
    #             raise ValueError("frequency_penalty must be between -2.0 and 2.0")
    #         model_kwargs['frequency_penalty'] = frequency_penalty
    #     if suffix:
    #         model_kwargs['suffix'] = suffix
    #     if model in chat_models:
    #         return ChatOpenAI(
    #             model=model,
    #             temperature=temperature or 0, 
    #             max_tokens=max_tokens,
    #             model_kwargs=model_kwargs,
    #             streaming=streaming,
    #         )
    #     else:    
    #         return OpenAI(
    #             model_name=model,
    #             temperature=temperature, 
    #             max_tokens=max_tokens,
    #             model_kwargs=model_kwargs,
    #             streaming=streaming,
    #         )
