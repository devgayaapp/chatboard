from typing import Optional
from pydantic import BaseModel
import pytest
import sys

from components.etl.prompt import prompt
sys.path.append('services/text_server')
from prompts.script_writer.script_writer_stream import script_writer_agent_stream, script_controller_prompt, script_scene_prompt



class JokeOutput(BaseModel):
    idea: str
    style: str
    lyrics: str


@prompt("script_writer/test", stream=True, pydantic_model=JokeOutput)
def joke_teller(output):
    return output

@pytest.mark.asyncio
async def test_joke_streaming():
    output = []
    async for message in joke_teller(topic="cats"):
        output.append(message)

    assert len(output) > 0
    final_output = output[-1].content
    assert len(final_output.idea) > 0
    assert 'idea' not in final_output.idea.lower()
    assert 'style' not in final_output.idea.lower()
    assert 'lyrics' not in final_output.idea.lower()
    
    assert len(final_output.style) > 0
    assert 'idea' not in final_output.style.lower()
    assert 'style' not in final_output.style.lower()
    assert 'lyrics' not in final_output.style.lower()

    assert len(final_output.lyrics) > 0
    assert 'idea' not in final_output.lyrics.lower()
    assert 'style' not in final_output.lyrics.lower()
    assert 'lyrics' not in final_output.lyrics.lower()




# class ScriptSceneOutput(BaseModel):
#     scene_type: str
#     idea: str
#     script: str


# @prompt("script_writer/scene_tool", stream=True, pydantic_model=ScriptSceneOutput)
# def script_scene_prompt(output):
#     return output

def is_surrounded_by_quotes(s):
    return len(s) >= 2 and s[0] == '"' and s[-1] == '"'


@pytest.mark.asyncio
async def test_script_scene_streaming():
    output = []

    async for message in script_scene_prompt(
        fact=None,
        scene_type="Hook",
        instructions="write a scene about cats",
        video_script=[],
        context="cats",
        agent_personality="sarcastic",
    ):
        output.append(message)
    assert len(output) > 0
    final_output = output[-1].content
    assert len(final_output.scene_type) > 0
    assert 'scene_type' not in final_output.scene_type.lower()
    assert 'idea' not in final_output.scene_type.lower()
    assert 'script' not in final_output.scene_type.lower()
    assert final_output.scene_type[0].isspace() == False
    assert not is_surrounded_by_quotes(final_output.scene_type)

    assert len(final_output.idea) > 0
    assert 'scene_type' not in final_output.idea.lower()
    assert 'idea' not in final_output.idea.lower()
    assert 'script' not in final_output.idea.lower()
    assert final_output.idea[0].isspace() == False
    assert not is_surrounded_by_quotes(final_output.idea)
    
    assert len(final_output.script) > 0
    assert 'scene_type' not in final_output.script.lower()
    assert 'idea' not in final_output.script.lower()
    assert 'script' not in final_output.script.lower()
    assert final_output.script[0].isspace() == False
    assert not is_surrounded_by_quotes(final_output.script)




# class ScriptControllerOutput(BaseModel):
#     thought: str
#     action: Optional[str]
#     action_input: Optional[str]
#     observation: Optional[str]
#     end_of_script: Optional[str]



# @prompt(
#     "script_writer/agent_controller",
#     stop_sequences=["Observation:", "Observation"],
#     stream=True,
#     pydantic_model=ScriptControllerOutput,
# )
# def script_controller_prompt(output):
#     return output



@pytest.mark.asyncio
async def test_script_controller_streaming():
    output = []
    async for message in script_controller_prompt(
                # facts=facts,
                prompt=prompt, 
                events=[], 
                tools=[],
                tool_names=[],                
                scene_types=["Hook", "Introduction", "Main Content", "Call-to-Action", "Conclusion", "Transition Scene"],
                thought_stream="",
                script_status="",                
                context="cats",
                agent_personality="sarcastic",
            ):
        output.append(message)

    assert len(output) > 0




@pytest.mark.asyncio
async def test_multiple_script_controller_streaming():
    for i in range(3):
        output = []
        async for message in script_controller_prompt(
                    # facts=facts,
                    prompt=prompt, 
                    events=[], 
                    tools=[],
                    tool_names=[],                
                    scene_types=["Hook", "Introduction", "Main Content", "Call-to-Action", "Conclusion", "Transition Scene"],
                    thought_stream="",
                    script_status="",                
                    context="cats",
                    agent_personality="sarcastic",
                ):
            output.append(message)
        assert len(output) > 10
        assert len(output) > 0



