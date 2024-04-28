# import re
# from openai import BaseModel
# import pytest
# from chatboard.etl.completion_parsing import parse_completion, auto_split_completion, sanitize_content, to_dict
# from logic.prompts.script_writer2.script_writer import script_writer_agent, write_hook_scene
# from chatboard.text.news_event import NewsEvent
# from chatboard.text.transcript import Transcript, Paragraph, Sentence


# events = [
#         NewsEvent.from_json(
#     {'date': '2024-02-11 12:06:02',
#     'text': 'An Analysis Of A Persistent Actors Activity\nDFIR researchers discovered an open directory containing over a years worth of historical threat actor activity. Through analysis of tools logs and artifacts exposed on the internet they were able to profile the threat actor and their targets. The research suggests that the primary motivation behind the threat actors actions was not financial despite occasional financially motivated behaviors such as deploying crypto-miners and targeting finance sites.\nThe threat actor consistently scanned government services and defense contractors for vulnerabilities but also exhibited limited financially driven activities. The threat actor exclusively relied on open source tools and frameworks including sqlmap and ghauri for active scanning and reconnaissance and Metasploit and Sliver for post-exploitation activities after exploiting vulnerabilities.\n',
#     'media': [],
#     'platform_id': 'None',
#     'platform': 'web',
#     'channel': 'cymulate',
#     'backend_id': 'eeb945217a05cb1c51f4338f05f1c9507dc84151c378eb31ed7429f8250de830',
#     'vector_uuid': None,
#     'content_type': None,
#     'topic': None})
# ]

# @pytest.mark.asyncio
# async def test_scene_writer():
#     topic = "cyber security"
#     instruction = f"write a short hook scene on {topic} based on the context"
#     message_list = []
#     async for message in write_hook_scene(
#         topic=topic,
#         events=events,
#         instruction=instruction
#     ):
#         # print(message.content)
#         message_list.append(message)


#     assert len(message_list) > 0
#     response = message_list[-1].prompt_response
#     assert response.value is not None
#     assert type(response.value) is Paragraph
#     assert len(response.value.text) > 100
#     s1 = set(topic.split(" "))
#     s2 = set(response.value.text.split(" "))
#     assert s1 & s2

    

# @pytest.mark.asyncio
# async def test_script_writer():
#     topic = "cyber security"
#     instruction = f"write a short script on {topic} based on the context"
#     message_list = []

#     async for message in script_writer_agent(
#         topic=topic,
#         events=events,
#         instruction=instruction
#     ):
#         message_list.append(message)

#     assert len(message_list) > 0
#     response = message_list[-1].agent_response
#     assert response.value is not None
#     assert type(response.value) is Transcript
