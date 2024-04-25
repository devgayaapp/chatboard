import re
from openai import BaseModel
import pytest
from components.etl.chat_agent import ChatAgent
from components.etl.chat_prompt import ChatPrompt
from dataclasses import Field
from langchain_core.pydantic_v1 import BaseModel, Field
from components.etl.chat_prompt import ChatPrompt
# from components.etl.system_conversation import ConversationRag, LangsmithConversationDataset
from components.etl.conversation import ConversationRag, LangsmithConversationDataset




@pytest.mark.asyncio
async def test_chat_inline_prompt():
    prompt = ChatPrompt(
    name="test_prompt",
    system_prompt="""
you should answer every question with the word "test" and nothing else.
""")
    
    question = "tell me about the weather"

    res = await prompt(f"""
answer the following question: {question}
""")
    assert "test" in res.value



@pytest.mark.asyncio
async def test_prompt_manager():
    prompt = ChatPrompt(
    "test_prompts/basic",
    name="basic_manager_prompt",    
    )
    question = "tell me about the weather"

    res = await prompt(f"""
answer the following question: {question}
""")
    assert "test" in res.value









class SwordTool(BaseModel):
    """this is a sword that the pirate can use to fight his enemies."""
    draw: bool = Field(..., description="true or false if the sword is drawn or not")
    sharp: bool = Field(..., description="true or false if the sword is sharp or not")
    length: float = Field(..., description="the length of the sword")


class GunTool(BaseModel):
    """this is a gun that the pirate can use to fight his enemies."""
    loaded: bool = Field(..., description="true or false if the gun is loaded or not")
    caliber: float = Field(..., description="the caliber of the gun")
    range: float = Field(..., description="the range of the gun")


class OfferMission(BaseModel):
    """this is a mission that the pirate can offer to the other pirates."""
    mission: str = Field(..., description="the mission that the pirate wants to offer to the other pirates")
    reward: float = Field(..., description="the reward that the pirate wants to offer to the other pirates")


class OfferDrink(BaseModel):
    """this is a drink that the pirate can offer to the other pirates."""
    drink: str = Field(..., description="the drink that the pirate wants to offer to the other pirates")
    price: float = Field(..., description="the price that the pirate wants to offer to the other pirates")



# tools = [get_tool_scheduler_prompt(convert_to_openai_tool(tool)) 
# for tool in [SwordTool, GunTool]]

@pytest.mark.asyncio
async def test_prompt_with_tools():
    prompt = ChatPrompt(
    "experimental/pirate",
    name="pirate_prompt",    
    )
    tool_response = await prompt.with_tools(prompt="draw your 20 inch sword and say something before the battle", tools=[SwordTool, GunTool])
    assert tool_response.tools is not None
    assert len(tool_response.tools) > 0
    assert type(tool_response.tools[0]) == SwordTool



@pytest.mark.asyncio
async def test_force_prompt_tool():
    prompt = ChatPrompt(
    "experimental/pirate",
    name="pirate_prompt",    
    )
    tool_response = await prompt.with_tools(
        prompt="draw your 20 inch sword and say something before the battle", 
        tool_choice="GunTool",
        tools=[SwordTool, GunTool]
    )
    assert tool_response.tools is not None
    assert len(tool_response.tools) > 0
    assert type(tool_response.tools[0]) == GunTool





def get_agent():
    agent = ChatAgent(
        "experimental/pirate_agent", 
        name="pirate_agent",
        tools=[SwordTool, GunTool, OfferMission, OfferDrink],
    )


    state = {
        "pirate":{
            "health": 100,
            "money": 100,
        },
        "visitor":{
            "health": 100,
            "money": 1000,
            "weapon": "sword",
        }
    }
    return agent, state


@pytest.mark.asyncio
async def test_agent_first_message():
    agent, state = get_agent()
    message = "hello there mate!"
    agent_response = await agent.next_step(
        message=message,
        **state
    )
    assert len(agent.conversation) == 3
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    assert agent_response.value == agent.conversation[2].content
    assert agent_response.conversation[0].content == agent.conversation[0].content
    assert agent_response.conversation[1].content == agent.conversation[1].content
    assert agent_response.conversation[2].content == agent.conversation[2].content




@pytest.mark.asyncio
async def test_agent_tool_use():
    agent, state = get_agent()
    message = "I am here to kill you!"
    agent_response = await agent.next_step(
        message=message,
        **state
    )
    assert len(agent.conversation) == 3
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    assert agent_response.value == agent.conversation[2].content
    assert agent_response.conversation[0].content == agent.conversation[0].content
    assert agent_response.conversation[1].content == agent.conversation[1].content
    assert agent_response.conversation[2].content == agent.conversation[2].content
    assert agent_response.tools is not None
    assert len(agent_response.tools) > 0
    assert type(agent_response.tools[0]) == SwordTool
    assert agent_response.conversation[-1].tool_calls is not None
    assert len(agent_response.conversation[-1].tool_calls) > 0
    assert type(agent_response.conversation[-1].tool_calls[0]) == SwordTool




@pytest.mark.asyncio
async def test_agent_multi_turn():
    agent, state = get_agent()
    message = "I am here to kill you!"
    agent_response = await agent.next_step(
        message=message,
        **state
    )
    assert len(agent.conversation) == 3
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    message = "I am here to kill you!"
    agent_response = await agent.next_step(
        message=message,
        **state
    )
    assert len(agent.conversation) == 5
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    assert agent.conversation[3].role == "user"
    assert agent.conversation[4].role == "assistant"
    assert agent_response.value == agent.conversation[4].content
    assert agent_response.conversation[3].content == agent.conversation[3].content
    assert agent_response.conversation[4].content == agent.conversation[4].content
    assert agent_response.tools is not None
    assert len(agent_response.tools) > 0
    assert type(agent_response.tools[0]) == SwordTool
    assert agent_response.conversation[-1].tool_calls is not None
    assert len(agent_response.conversation[-1].tool_calls) > 0
    assert type(agent_response.conversation[-1].tool_calls[0]) == SwordTool

    message = "is was a joke! just give me a mission"
    agent_response = await agent.next_step(
        message=message,
        **state
    )
    assert len(agent.conversation) == 7
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
    assert agent.conversation[2].role == "assistant"
    assert agent.conversation[3].role == "user"
    assert agent.conversation[4].role == "assistant"
    assert agent.conversation[5].role == "user"
    assert agent.conversation[6].role == "assistant"

    
    assert agent_response.conversation[-3].tool_calls is not None
    assert len(agent_response.conversation[-3].tool_calls) > 0
    assert type(agent_response.conversation[-3].tool_calls[0]) == SwordTool

    assert len(agent_response.tools) > 0
    assert type(agent_response.tools[0]) == OfferMission
    assert agent_response.conversation[-1].tool_calls is not None
    assert len(agent_response.conversation[-1].tool_calls) > 0
    assert type(agent_response.conversation[-1].tool_calls[0]) == OfferMission






@pytest.mark.asyncio
async def test_rag_prompt():
    test_segmentizer = ChatPrompt(
        "test_prompts/test_segmentizer", 
        name="test_segmentizer", 
        rag_length=3
    )
    segmentizer_rag = ConversationRag("test_segmentizer")
    dataset = LangsmithConversationDataset("test_segmentizer")

    await segmentizer_rag.delete_all()

    sentence="queen Elizabeth II of England, had a networth of 1000000$ in 1967, and had 4 children with prince Philip, duke of edinburgh."
    
    # sentence = "the s&p 500 increased by 10% in Q2 of 2020."
    # sentence = "Adolf Hitler, was the leader of germany, during WW2."
    # sentence = "Apple is known for the iPhone, iPad, and Macbook."
    # sentence = "ООН была создана 24 октября 1945 года."
    output = await test_segmentizer(sentence=sentence)
    


    all_examples = await segmentizer_rag.get_many()

    assert len(all_examples) == 0

    conversation_list = dataset.get_records(is_agent=False)    
    examples_list = conversation_list[:1]

    add_response = await segmentizer_rag.add_conversation_list(examples_list)

    assert len(add_response['records']) == len(examples_list)
    assert add_response['records'][0]['id'] == examples_list[0].id

    all_examples = await segmentizer_rag.get_many()

    assert len(all_examples) == 1


    sentence = "the F-16, was invented in the USA in 1974, by General Dynamics."
    output = await test_segmentizer(sentence=sentence)
    assert output.examples is not None
    assert len(output.examples) == 1

    examples_list = conversation_list[:1]
    add_response = await segmentizer_rag.add_conversation_list(examples_list)

    assert len(add_response['records']) == len(examples_list)
    assert add_response['records'][0]['id'] == examples_list[0].id


    examples_list = conversation_list
    add_response = await segmentizer_rag.add_conversation_list(examples_list)

    assert len(add_response['records']) == len(examples_list)
    assert add_response['records'][0]['id'] == examples_list[0].id
    assert add_response['records'][1]['id'] == examples_list[1].id

    all_examples = await segmentizer_rag.get_many()

    assert len(all_examples) == 2


    sentence = "the s&p 500 increased by 10% in Q2 of 2020."
    output = await test_segmentizer(sentence=sentence)
    assert output.examples is not None
    assert len(output.examples) == 2

    await segmentizer_rag.delete_all()
