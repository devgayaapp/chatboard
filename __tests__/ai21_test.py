# import pytest
# from components.apis.ai21 import generate_paragraphs_prompt, generate_paragraph_prompt_text, get_section_prompts, get_section_prompt_text


# def test_section_prompts():
#     prompt1 = get_section_prompts(9)
#     assert len(prompt1) > 0
#     prompt2 = get_section_prompt_text('good title')
#     assert len(prompt2) > 0
#     prompt1+=prompt2
#     assert prompt1[-1] == ':'
    
#     with pytest.raises(ValueError):
#         get_section_prompts(1)

#     with pytest.raises(ValueError):
#         get_section_prompts(-1)

#     with pytest.raises(ValueError):
#         get_section_prompts(9000)


# def test_paragraph_prompts():

#     prompt1 = generate_paragraphs_prompt(
#         total_sections = 3,
#         section_num = 2,
#         samples = 2,
#     )
#     assert len(prompt1) > 0

#     prompt2 = generate_paragraph_prompt_text('good title', '1. section1\n2. section2\n3. section3\n', 2, 'section2')
#     assert len(prompt2) > 0

#     prompt1 += prompt2
#     assert prompt1[-1] == ':'

#     with pytest.raises(ValueError):
#         generate_paragraphs_prompt(1, 1)

#     with pytest.raises(ValueError):
#         generate_paragraphs_prompt(2, 4)

#     with pytest.raises(ValueError):
#         generate_paragraphs_prompt(-4, 2)

#     with pytest.raises(ValueError):
#         generate_paragraphs_prompt(4, -2)



# def test_paragraph_prompts_exhaustion():    
#     for i in range(2, 41):
#         for j in range(1, i):
#             prompt = generate_paragraphs_prompt(
#                 total_sections = i,
#                 section_num = j,
#                 samples = 2,
#             )
#             assert len(prompt) > 0