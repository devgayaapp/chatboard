from langchain_core.utils.function_calling import convert_to_openai_tool





from pydantic import BaseModel


class AppManager:

    def __init__(self):
        self.rag_spaces = {}
        self.prompts = {}


    def register_rag_space(self, namespace: str, metadata_class: BaseModel):
        if namespace in self.rag_spaces:
            return
        self.rag_spaces[namespace] = {
            "metadata_class": metadata_class,
            "namespace": namespace,
        }
    
    def register_prompt(self, prompt):
        self.prompts[prompt.name] = prompt


    def get_metadata(self):
        rag_space_json = [{
            "namespace": namespace,
            "metadata_class": convert_to_openai_tool(rag_space["metadata_class"])
        } for namespace, rag_space in self.rag_spaces.items()]

        return {
            "rag_spaces": rag_space_json,            
        }
    
    # def get_rag_manager(self, namespace: str):
    #     rag_cls = self.rag_spaces[namespace]["metadata_class"]
    #     ns = self.rag_spaces[namespace]["namespace"]
    #     return RagDocuments(ns, rag_cls)

app_manager = AppManager()