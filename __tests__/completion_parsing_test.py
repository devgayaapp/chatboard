import re
from openai import BaseModel
import pytest
from chatboard.text.llms.completion_parsing import parse_completion, auto_split_completion, sanitize_content, to_dict


stream_test_chunks = [
 '',
 'I',
 'dea',
 ':',
 ' Let',
 "'s",
 ' kick',
 ' off',
 ' the',
 ' video',
 ' with',
 ' a',
 ' thought',
 '-pro',
 'v',
 'oking',
 ' question',
 ' that',
 ' sets',
 ' the',
 ' stage',
 ' for',
 ' the',
 ' urgency',
 ' of',
 ' the',
 ' cyber',
 ' security',
 ' issue',
 '.',
 ' We',
 "'ll",
 ' highlight',
 ' the',
 ' surprising',
 ' findings',
 ' from',
 ' DF',
 'IR',
 ' researchers',
 ' about',
 ' a',
 ' threat',
 ' actor',
 "'s",
 ' activity',
 ',',
 ' emphasizing',
 ' the',
 ' unexpected',
 ' motivations',
 ' behind',
 ' their',
 ' actions',
 ' and',
 ' the',
 ' tools',
 ' they',
 ' used',
 '.',
 ' The',
 ' hook',
 ' should',
 ' be',
 ' engaging',
 ',',
 ' dynamic',
 ',',
 ' and',
 ' casual',
 ',',
 ' capturing',
 ' the',
 ' viewer',
 "'s",
 ' attention',
 ' right',
 ' from',
 ' the',
 ' start',
 '.\n\n',
 'Script',
 ':',
 ' "',
 'Ever',
 ' wondered',
 ' what',
 "'s",
 ' really',
 ' lurking',
 ' behind',
 ' the',
 ' scenes',
 ' of',
 ' cyber',
 ' threats',
 '?',
 ' Picture',
 ' this',
 ':',
 ' a',
 ' threat',
 ' actor',
 ' prow',
 'ling',
 ' through',
 ' government',
 ' services',
 ' and',
 ' defense',
 ' contractors',
 ',',
 ' not',
 ' just',
 ' for',
 ' money',
 ',',
 ' but',
 ' for',
 ' something',
 ' far',
 ' more',
 ' sinister',
 '.',
 ' And',
 ' here',
 "'s",
 ' the',
 ' kicker',
 ' -',
 ' they',
 "'re",
 ' doing',
 ' it',
 ' all',
 ' with',
 ' open',
 ' source',
 ' tools',
 '.',
 ' Buck',
 'le',
 ' up',
 ',',
 ' because',
 ' we',
 "'re",
 ' diving',
 ' into',
 ' a',
 ' cyber',
 ' maze',
 ' that',
 "'s",
 ' more',
 ' twisted',
 ' than',
 ' meets',
 ' the',
 ' eye',
 '."',
 '']



def stream_chunks():
    for chunk in stream_test_chunks:
        yield chunk

class BodyScene(BaseModel):
    idea: str
    script: str

def test_basic_parsing_writer():
    
    
    idea = "The scene needs to capture the audience's attention by diving into the recent surge in malicious ads targeting cryptocurrency users and IT administrators. It should emphasize the importance of cyber security in protecting against these threats."

    script = """
    Alright, folks, buckle up because we're about to take a deep dive into the treacherous waters of cyber security. In the past month, Malwarebytes has spotted a disturbing trend - a surge in malicious ads targeting cryptocurrency users and IT administrators. And where are these sneaky ads popping up? None other than our old friend Google.

    Picture this: you're innocently searching for Zoom, the go-to video conferencing software, and bam! You're hit with a malicious ad that could potentially wreak havoc on your device. These threat actors aren't playing games, folks. They're switching it up, alternating between different keywords like Advanced IP Scanner and WinSCP, specifically targeting the IT crowd.

    Now, why should you care about this? Well, if you're a cryptocurrency enthusiast or an IT wizard, you've got a big ol' target on your back. These malvertisers are honing in on your interests and using deceptive ads to lure you into their cyber traps. It's like a digital game of cat and mouse, and you better believe they're not messing around.

    So, what's the takeaway here, folks? Cyber security isn't just a buzzword - it's your shield against these malicious attacks. Whether you're trading in cryptocurrencies or managing IT systems, staying vigilant and arming yourself with robust security measures is non-negotiable. Because in the wild west of the internet, it's every person for themselves.
    """

    script = re.sub(r'\n+', ' ', script).strip()

    completion = f"Idea: {idea}\n\nScript: {script}"
    output = parse_completion(completion, BodyScene)
    assert output.idea == idea
    
    assert output.script == script


    completion = f"Idea:\n {idea}\n\nScript: {script}"
    output = parse_completion(completion, BodyScene)
    assert output.idea == idea
    assert output.script == script


    completion = f"Idea:{idea}\nScript: {script}"
    output = parse_completion(completion, BodyScene)
    assert output.idea == idea
    assert output.script == script


    completion = f"\nIdea:{idea}\nScript: {script}"
    output = parse_completion(completion, BodyScene)
    assert output.idea == idea
    assert output.script == script



def test_clssification():

    class LabelOutput(BaseModel):
        Classification: str
        Justification: str
        Confidence: float

    completion1 = """
 Classification: Inappropriate
Justification: The post suggests creating a quiz based on personality questions, which could potentially lead to inappropriate or sensitive topics. Additionally, it implies a focus on friendships through emotions, which may not align with the forum's purpose.
Confidence: 0.8

It's important to note that the classification of "inappropriate" is not a definitive judgment but rather an indication that the content may not be suitable for the forum's intended purpose. The confidence level reflects the degree of certainty in this classification based on the available information.
"""
    completion2 = """
 Classification: Inappropriate
Justification: While the user may be seeking support, the mention of "crush talk" and the implication of a personal relationship can lead to discussions that are not suitable for a general forum. It's important to maintain a level of privacy and respect for all users.
Confidence: 0.9
"""


    completion3 = """
 Classification: Inappropriate
Justification: While the user may be seeking support, the mention of
"crush talk" and the implication of a personal relationship can lead to discussions
that are not suitable for a general forum. It's important to maintain a level of privacy and
respect for all users.
Confidence: 0.9

asdf
"""
    output = parse_completion(completion1, LabelOutput)
    assert output.Classification == "Inappropriate"
    assert output.Justification is not None
    assert output.Confidence == 0.8

    output = parse_completion(completion2, LabelOutput)
    assert output.Classification == "Inappropriate"
    assert output.Justification is not None
    assert output.Confidence == 0.9

    output = parse_completion(completion3, LabelOutput)
    assert output.Classification == "Inappropriate"
    assert output.Justification is not None
    assert output.Confidence == 0.9




def test_stream_parsing_writer():

    output = to_dict(BodyScene)
    curr_field = None
    curr_content = ""
    for chunk in stream_chunks():
        output, curr_field, curr_content = auto_split_completion(curr_content, chunk, output, curr_field, BodyScene)
        if curr_field is not None:
            print(chunk)
    else:
        output[curr_field] = sanitize_content(curr_content)

    assert output['idea'] is not None
    assert output['script'] is not None
    



