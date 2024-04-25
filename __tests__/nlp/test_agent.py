import pytest
from ray import serve
import ray
from graph_repo.utils import ChatHistory
from graph_repo.voice_pipe import TextGeneratorResources, deployment_graph, SendMessagePrompt



def test_basic_flow():
    chat = ChatHistory("Syndy", "you are a helpfull bot. do not reply to this message")
    handler = serve.run(deployment_graph)
    request = {
        'chat': chat.to_json(), 
        'voice':'v2/en_speaker_0'
    }
    ref = handler.remote(request)
    res = ray.get(ref)
    print(res)


@pytest.mark.asyncio
async def test_message_prompt():
    # llm = TextGeneratorResources()
    handler = serve.run(SendMessagePrompt.bind(TextGeneratorResources.bind(), '__tests__/nlp/test_data/basic_prompt.yaml'))
    res_ref = await handler.complete.remote(text="plants")
    # res_ref = handler.complete_sync.remote(text="hello")
    res = ray.get(res_ref)
    print(res)
     