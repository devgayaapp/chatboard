import pytest
import json
from components.text.nlp.image_description import generate_image_idea_from_sentence2

from components.text.transcript import Transcript





def test_generate_image_idea_from_sentence():

    with open('__tests__/data/simple_content.json') as f:
        content = json.load(f)
        transcript = Transcript.from_lexical(content)

    in_sentence = transcript[0][0]
    topic = "why cats are the best"
    ideas_sentence = generate_image_idea_from_sentence2(in_sentence, topic)
    print(ideas_sentence)
    