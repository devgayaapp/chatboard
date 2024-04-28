import pytest

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import Field

from chatboard.text.llms.prompt import Prompt
from chatboard.text.llms.llm import LLM, OpenAiLlmClient, PhiLLM, OpenAILLM




@pytest.mark.asyncio
async def test_chat_prompt_class():
    class TestPrompt(Prompt):
        name: str = "test_prompt"
        is_traceable: bool = False
        # llm: PhiLLM = Field(default_factory=PhiLLM)
        llm: OpenAILLM = Field(default_factory=OpenAILLM)
        system_prompt: str = """
        you should say "yes" for every question
        """
        # async def render_system_prompt(self, **kwargs):
        #     return f"""
            
        #     """

        async def render_prompt(self, **kwargs):
            return f"""
            tell me a story about a pirate
            """

    test_prompt = TestPrompt()
    msg = await test_prompt()
    assert "yes" in msg.content.lower()



@pytest.mark.asyncio
async def test_chat_inline_prompt():
    class TestPrompt(Prompt):
        name: str = "test_prompt"
        is_traceable: bool = False
        llm: PhiLLM = Field(default_factory=PhiLLM)
        system_prompt: str = """
        you should say "yes" for every question
        """    

    test_prompt = TestPrompt()
    msg = await test_prompt("what is your name?")
    assert "yes" in msg.content.lower()

    class TestPrompt(Prompt):
        name: str = "test_prompt"
        is_traceable: bool = False
        llm: OpenAILLM = Field(default_factory=OpenAILLM)
        system_prompt: str = """
        you should say "yes" for every question
        """    

    test_prompt = TestPrompt()
    msg = await test_prompt("what is your name?")
    assert "yes" in msg.content.lower()




