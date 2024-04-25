import sys
from typing import List
from uuid import uuid4

from components.etl.conversation import AIMessage, ConversationRagMetadata, HumanMessage, RagConversationVectorizer
sys.path.append("logic")


import json

from pydantic import BaseModel
from components.etl.rag_manager import RagVectorSpace, RagVector
# from components.etl.rag_manager import RAGMetadata
import pytest

from logic.resources.state_embeddings import RagStateViewVectorizer, StateViewMetadata


# class NerRagMetadata(BaseModel):
#     language: str
#     key: List[HumanMessage]
#     value: List[AIMessage]




@pytest.mark.asyncio
async def test_basic_operations():
    vectorizer = RagConversationVectorizer()
    rag_manager = RagVectorSpace('test_rag', vectorizer, ConversationRagMetadata)
    try:
        # rag_example = {
        #     "id": str(uuid4()),
        #     "key": [HumanMessage(content="test")],
        #     "text": [AIMessage(content="this is the return text")],
        #     "language":"en"
        # }

        rag_example = RagVector[ConversationRagMetadata](
            id=str(uuid4()),
            metadata=ConversationRagMetadata(
                inputs=[HumanMessage(content="test this")],
                output=AIMessage(content="this is the return text")
            )
        )
        rag_example.metadata.inputs
        add_res = await rag_manager.add_many([rag_example])
        assert add_res is not None
        assert add_res['records'][0]['id'] == rag_example.id
        add_res_inputs = json.loads(add_res['records'][0]['metadata']['inputs'])
        add_res_outputs = json.loads(add_res['records'][0]['metadata']['output'])
        assert add_res_inputs[0]['content'] == rag_example.metadata.inputs[0].content
        assert add_res_outputs['content'] == rag_example.metadata.output.content
        # assert add_res.key == rag_example.metadata.key
        # assert add_res.metadata.key == rag_example.metadata.key
        # assert add_res.metadata.value == rag_example.metadata.value
        # assert add_res.metadata.language == rag_example["language"]

        example_list = await rag_manager.get_many()
        assert len(example_list) == 1
        assert example_list[0] is not None
        assert example_list[0].id == rag_example.id
        assert example_list[0].metadata.inputs[0].content == rag_example.metadata.inputs[0].content
        assert example_list[0].metadata.output.content == rag_example.metadata.output.content


        search_list = await rag_manager.similarity([HumanMessage(content="test")])
        assert len(search_list) == 1
        assert search_list[0] is not None
        assert example_list[0].metadata.inputs[0].content == rag_example.metadata.inputs[0].content
        assert example_list[0].metadata.output.content == rag_example.metadata.output.content


        rag_example2 = RagVector[ConversationRagMetadata](
            id=str(uuid4()),
            metadata=ConversationRagMetadata(
                inputs=[HumanMessage(content="cats are great")],
                output=AIMessage(content="yeah I know right?")
            )
        )
        add_res = await rag_manager.add_many([rag_example2])

        example_list = await rag_manager.get_many()
        assert len(example_list) == 2

        search_list = await rag_manager.similarity([HumanMessage(content="test")])
        assert len(search_list) == 2
        assert search_list[0] is not None
        assert example_list[0].metadata.inputs[0].content == rag_example.metadata.inputs[0].content
        assert example_list[0].metadata.output.content == rag_example.metadata.output.content

    finally:
        await rag_manager.delete_all()




from logic.prompts.media_editor_agent5.positioning import Placement
from logic.prompts.media_editor_agent5.tools import DeleteElement
from logic.prompts.media_editor_agent5.state import StockClipState, CutOutState, TextOverlayState, BackgroundPlateState
from logic.prompts.media_editor_agent5.elements import BackgroundPlate, TextOverlay, CutOut, StockClip
from logic.resources.state_embeddings import RagStateViewVectorizer, StateViewMetadata
from components.etl.rag_manager import RagVectorSpace
from components.etl.prompt_tracer import PromptTracer



@pytest.mark.asyncio
async def test_state_view_rag():

    state_options = [StockClipState, CutOutState, TextOverlayState, BackgroundPlateState]
    tool_options = [Placement, StockClip, CutOut, TextOverlay, BackgroundPlate, DeleteElement]
    vectorizer = RagStateViewVectorizer(state_options, tool_options)


    rag_space = RagVectorSpace(
            "test_state_view",
            vectorizer,
            StateViewMetadata,
        )

    try:


        
        await rag_space.delete_all()



        example_data = json.load(open("components/__tests__/data/rag/rag_state_view.json", "r"))
        example0 = example_data[0]
        example1 = example_data[1]
        example2 = example_data[2]
        example3 = example_data[3]

        embs1 = await rag_space.vectorizer.embed_query(RagVector[StateViewMetadata](**example1).metadata)
        embs1
        embs2 = await rag_space.vectorizer.embed_query(RagVector[StateViewMetadata](**example2).metadata)
        embs22 = await rag_space.vectorizer.embed_query(RagVector[StateViewMetadata](**example2).metadata)
        
        assert embs1.dense == embs2.dense
        assert embs1.sparse != embs2.sparse
        assert embs2.sparse == embs22.sparse

        await rag_space.add_many([example3])

        saved_examples = await rag_space.get_many()
        len(saved_examples) == 1

        similar0 = await rag_space.similarity(StateViewMetadata(**example0["metadata"]), alpha=0.5)
        similar1 = await rag_space.similarity(StateViewMetadata(**example1["metadata"]), alpha=0.5)
        similar2 = await rag_space.similarity(StateViewMetadata(**example2["metadata"]), alpha=0.5)

        assert similar0[0].score < similar1[0].score
        assert similar1[0].score < similar2[0].score
    finally:
        await rag_space.delete_all()
        