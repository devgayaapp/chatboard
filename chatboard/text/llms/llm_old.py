from langchain.llms.base import LLM
from typing import Any, List, Optional, TypeVar
from langchain.callbacks.manager import CallbackManagerForLLMRun



class CustomLLM(LLM):

    llm_runnable: Any

    def __init__(self, llm_runnable):
        super().__init__()  # Call the parent class's constructor if necessary
        self.llm_runnable = llm_runnable

    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any):        
        output = self.llm_runnable.complete(prompt, stop)
        return output

    async def _acall(self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any):
        output = await self.llm_runnable.complete.run_async(prompt, stop)
        return output