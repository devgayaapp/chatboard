from typing import List, Type
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool







def parse_properites(properties, add_type=True, add_constraints=True, tabs="\t"):
    prompt = ""
    for prop, value in properties.items():
        param_promp = f"\n{tabs}{prop}"
        if 'allOf' in value: 
            obj = value['allOf'][0]
            prompt += f"\n{tabs}{obj['title']}:"
            prompt += parse_properites(obj['properties'], tabs=tabs+"\t")
        else:
            if add_type:
                param_promp += f":({value['type']})"
            if 'description' in value:
                param_promp += f" {value['description']}"
            if add_constraints and ('minimum' in value or 'maximum' in value):
                param_promp += f". should be"
                if 'minimum' in value:
                    param_promp += f" minimum {value['minimum']}"
                if 'maximum' in value:
                    param_promp += f" maximum {value['maximum']}"
                param_promp += "."
            prompt += param_promp
    return prompt



def model_to_prompt(tool_dict, add_type=True, add_constraints=True):    
    tool_function = tool_dict['function']
    prompt = f"""{tool_function["name"]}:"""
    properties = tool_dict['function']["parameters"]['properties']
    prompt += parse_properites(properties, add_type, add_constraints)
    return prompt 




class ViewModel(BaseModel):

    def render(self):
        return self.dict()
    
    @classmethod    
    def render_system(self):
        return model_to_prompt(convert_to_openai_tool(self))
    
    def vectorize(self):
        return None
    
    @classmethod 
    def to_tool(self):
        return convert_to_openai_tool(self)


class Action(BaseModel):

    async def reduce(self, state: ViewModel):
        return None

    @classmethod    
    def render(self):
        return model_to_prompt(convert_to_openai_tool(Action))

