# import sys
# sys.path.append("logic")

# import re
# import numpy as np
# from openai import BaseModel
# import pytest


# from logic.resources.state_embeddings import FrameEncoder, ToolSparseEncoder
# from logic.prompts.media_editor_agent6.evaluations import create_test_state
# from logic.prompts.media_editor_agent6.positioning import Placement
# from logic.prompts.media_editor_agent6.state import StockClipState, CutOutState, TextOverlayState, BackgroundPlateState
# from logic.prompts.media_editor_agent6.elements import BackgroundPlate, TextOverlay, CutOut, StockClip
# from logic.prompts.media_editor_agent6.tools import DeleteElement





# def test_action_encoder():
#     max_frames = 2
#     max_layers = 2
#     max_iterations = 3

#     state_options = [StockClipState]
#     tool_options = [Placement, StockClip]

#     tool_encoder = ToolSparseEncoder(tool_options, max_iterations)
#     state, actions = create_test_state([
#             ("TO", [1], 0),
#             ("CO", [1], 1),
#             # ("P", 1, [1], 1),    
#         ], frames=2)

#     mat = tool_encoder.transform([actions])
#     assert(np.any(mat.toarray()) == False)

#     # state, actions = create_test_state([
#     #         ("SC", 1),
#     #         ("P", 1, [0], 0),
#     #     ], frames=2)

#     def evaluate_action_vector(actions, tool_options, max_iterations):
#         len_tools = len(tool_options)
#         tool_encoder = ToolSparseEncoder(tool_options, max_iterations)
#         mat = tool_encoder.transform([actions])
#         assert(np.any(mat.toarray()) == True)
#         assert(mat.toarray().shape == (1, max_iterations * len_tools))
#         for i in range(max_iterations):
#             assert np.sum(mat[0,(i*len_tools):(i+1)*len_tools].toarray()) <= 1
#         # assert(mat.toarray()[0, 0])
#         mat.toarray()


#     # state1, actions1 = create_test_state([
#     #     ("SC", 1),
#     #     ("P", 1, [0,1], 0),
#     #     ("TO", 2),
#     #     ("P", 2, [1], 1),
#     # ], frames=2)
#     test_state_data = [
#         create_test_state([
#             ("SC", [0], 1),
#         ], frames=2),
#         create_test_state([
#             ("TO", [1], 1),  
#         ], frames=2),
#         create_test_state([
#             ("SC", [0, 1, 2], 0),
#             ("SC", [3, 4], 0),
#             ("TO", [3, 4], 1),
#         ], frames=5),
#     ]


#     max_iterations = 6

#     state_options = [StockClipState, CutOutState, TextOverlayState, BackgroundPlateState]
#     tool_options = [Placement, StockClip, CutOut, TextOverlay, BackgroundPlate, DeleteElement]

#     for state, actions in test_state_data:
#         evaluate_action_vector(actions, tool_options, max_iterations)




# def test_animation_position_encoder():
    




# def test_frame():

#     max_frames = 2
#     max_layers = 2
#     max_iterations = 3

#     state_options = [StockClipState]
#     tool_options = [Placement, StockClip]


#     state, actions = create_test_state([
#             ("TO", 1),
#             ("CO", 1),
#             # ("P", 1, [1], 1),    
#         ], frames=2)

#     frame_encoder = FrameEncoder(
#             state_options, 
#             max_frames=max_frames, 
#             max_layers=max_layers
#         )

#     mat = frame_encoder.transform([state.elements])

#     assert(np.any(mat.toarray()) == False)

#     test_state_data = [
#         create_test_state([
#             ("SC", 1),
#             ("P", 1, [0], 0),    
#         ], frames=2),
#         create_test_state([
#             ("TO", 1),
#             ("P", 1, [1], 1),    
#         ], frames=2),
#         create_test_state([
#             ("SC", 1),
#             ("P", 1, [0, 1, 2], 0), 
#             ("SC", 2),
#             ("P", 2, [3, 4], 0), 
#             ("TO", 3),
#             ("P", 3, [3, 4], 1), 
#         ], frames=5),
#     ]


#     max_iterations = 6
#     max_frames = 5
#     max_layers = 2

#     state_options = [StockClipState, CutOutState, TextOverlayState, BackgroundPlateState]
#     tool_options = [Placement, StockClip, CutOut, TextOverlay, BackgroundPlate, DeleteElement]

#     frame_encoder = FrameEncoder(state_options, max_frames, max_layers)

#     for state, actions in test_state_data:
#         len_state = len(state_options)
#         mat = frame_encoder.transform([state.elements])
#         assert(np.any(mat.toarray()) == True)
#         assert(mat.toarray().shape == (1, len_state * max_frames * max_layers))
#         for i in range(len_state * max_frames * max_layers):
#             assert np.sum(mat[0,(i*len_state):(i+1)*len_state].toarray()) <= 1




