import json
import pytest
import sys
from components.audio.audio import Audio


module_path = 'voice_server'
sys.path.insert(0, module_path)
from data.testing_data.lexical_test_data import lexical_data
from components.text.transcript import Paragraph, Sentence, Transcript, Word
from config import AWS_VOICE_BUCKET, TEMP_DIR, DATA_DIR, TESTING_DATA_DIR
from util.sanitizers import get_sanitized_filename





def split_text_generator(text, start_time=0, char_duration = 1):
    curr_time = start_time
    for para in [p for p in text.split('\n') if p != '']:
        words = [w for w in para.split(' ') if w != '']
        for i, w in enumerate(words):
            duration = len(w) * char_duration 
            w = w+' ' if i != len(words)-1 else w+'\n'
            yield w, curr_time, duration
            curr_time += duration + 1



def test_lexical_state_processing():

    transcript = Transcript.from_lexical(lexical_data)

    # assert len(transcript) == 16

    content = transcript.to_lexical()


    for pi in range(len(transcript.paragraphs)):
        assert len(content['root']['children'][pi]['children']) == len(lexical_data['root']['children'][pi]['children'])
        for si in range(len(transcript[pi].sentences)):
            assert len(set(content['root']['children'][pi]['children'][si].keys()) - set(lexical_data['root']['children'][pi]['children'][si].keys())) == 0

            for field in ['duration', 'time', 'isDirty', 'uuid', 'audio', 'audioText', 'isAiGenerated']:
                assert content['root']['children'][pi]['children'][si].get(field, None) == lexical_data['root']['children'][pi]['children'][si].get(field, None)

            assert len(content['root']['children'][pi]['children'][si]['children']) == len(lexical_data['root']['children'][pi]['children'][si]['children'])
            for wi in range(len(transcript[pi][si])):

                for field in ['text', 'time', 'duration', 'score', 'entity', 'sentiment', 'phraseIndex', 'color', 'backgroundColor']:
                    assert content['root']['children'][pi]['children'][si]['children'][wi].get(field, None) == lexical_data['root']['children'][pi]['children'][si]['children'][wi].get(field, None)





def test_word_functions():
    word = Word('dog ')
    assert word.text == 'dog '
    # assert word.time == None
    # assert word.duration == None
    assert word.is_end_of_sentence == False
    assert word.is_end_of_paragraph == False

    word = Word('dog ', 10, 1)
    assert word.text == 'dog '
    # assert word.time == 10
    # assert word.duration == 1
    assert word.is_end_of_sentence == False
    assert word.is_end_of_paragraph == False

    word = Word('dog. ')
    assert word.is_end_of_sentence == True
    word = Word('dog.')
    assert word.is_end_of_sentence == True
    word = Word('dog? ')
    assert word.is_end_of_sentence == True
    word = Word('dog! ')
    assert word.is_end_of_sentence == True
    word = Word('dog!')
    assert word.is_end_of_sentence == True
    word = Word('dog  ! ')
    assert word.is_end_of_sentence == True

    word = Word('dog. \n')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True

    word = Word('dog. \n')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True

    word = Word('dog.\n')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True

    word = Word('dog.\n ')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True

    word = Word('dog.      \n')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True

    word = Word('dog\n')
    assert word.is_end_of_sentence == True
    assert word.is_end_of_paragraph == True





def test_appending_words_to_sentence():

    text = "This is a single sentence."
    char_duration = 1

    sent = Sentence()
    
    # curr_time = 0
    # for w in text.split():
        # duration = len(w) * char_duration 
        # word = Word(w+' ', curr_time, duration)
        # curr_time += duration + 1
        # sent.append(word)
    for w, time, duration in split_text_generator(text):
        word = Word(w, time, duration)
        sent.append(word)

    assert len(sent) == 5
    assert sent.duration == 26


    sent2 = Sentence(time=0, duration=30)

    # curr_time = 0
    # for w in text.split():
    #     duration = len(w) * char_duration 
    #     word = Word(w+' ', curr_time, duration)
    #     curr_time += duration + 1
    #     sent2.append(word)
    for w, time, duration in split_text_generator(text):
        word = Word(w, time, duration)
        sent2.append(word)


    assert len(sent2) == 5
    assert sent2.duration == 30


def test_sentence_from_lexical():
    lex_sentence = None
    with open('__tests__/data/simple_content.json', 'r') as f:
        lexical_data = json.load(f)
        print(lexical_data)
        lex_sentence = lexical_data['root']['children'][0]['children'][0]

    sent = Sentence.from_lexical(lex_sentence)
    assert sent.uuid == lex_sentence['uuid']
    assert len(sent.words) == len(lex_sentence['children'])
    for i in range(len(sent.words)):
        assert sent.words[i].text == lex_sentence['children'][i]['text']
    


def test_appending_sentences_to_paragraphs():
    text = "This is a single sentence. "

    para = Paragraph()
    # for i in range(4):
    #     sent = Sentence()
    #     for w, time, duration in split_text_generator(text, start_time=10):
    #         word = Word(w, time, duration)
    #         sent.append(word)
    sent = Sentence(text)
    para.append_sentence(sent)

    assert len(para.sentences) == 1

    para.append_sentence()

    assert len(para.sentences) == 2

    sent = Sentence(text)
    para.append_sentence(sent)

    assert len(para.sentences) == 3

    

def testing_appending_paragraphs_and_sentences_to_transcript():
        
    transcript = Transcript() 

    transcript.append_paragraph(Paragraph())

    assert(len(transcript.paragraphs) == 1)

    transcript.append_paragraph()

    assert(len(transcript.paragraphs) == 2)

    transcript.append_paragraph(Paragraph())

    assert(len(transcript.paragraphs) == 3)


def test_appending_words_to_transcript():
    single_sentence_text = "This is a single sentence."

    transcript1 = Transcript()

    for w in single_sentence_text.split():
        word = Word(w+' ')
        transcript1.append_word(word)

    assert len(transcript1) == 5
    assert len(transcript1[0].sentences) == 1
    assert len(transcript1.paragraphs) == 1

    multi_sentence_text = "This is first sentence. This is second sentence. This is third sentence."

    transcript2 = Transcript()
    for w in multi_sentence_text.split():
        word = Word(w+' ')
        transcript2.append_word(word)

    assert len(transcript2) == 12
    assert len(transcript2[0].sentences) == 3
    assert len(transcript2.paragraphs) == 1

    tracks = [
        {'speaker': 0, 'text': "This is first sentence"},
        {'speaker': 1, 'text': "This is second sentence."},
        {'speaker': 0, 'text': "This is third sentence"},
        {'speaker': 2, 'text': "This is forth sentence."},
    ]

    transcript3 = Transcript()

    for i, track in enumerate(tracks):
        for w in track['text'].split():
            word = Word(w+' ')
            transcript3.append_word(word, speaker=track['speaker'])
    
    assert len(transcript3) == 16
    assert len(transcript3.paragraphs) == 4
    # assert len(transcript3[0].sentences) == 4
    assert len(transcript3.unique_speakers()) == 3
    assert transcript3[0].speaker == 0
    assert transcript3[0][0].speaker == 0
    assert transcript3[1].speaker == 1
    assert transcript3[1][0].speaker == 1    
    assert transcript3[2].speaker == 0
    assert transcript3[2][0].speaker == 0
    assert transcript3[3].speaker == 2
    assert transcript3[3][0].speaker == 2
    

        


def test_sentence_abs_time():
    text = "This is a single sentence."
    sent = Sentence()
    for w, time, duration in split_text_generator(text, start_time=10):
        word = Word(w, time, duration)
        sent.append(word)

    assert sent.time == 10
    assert sent[0].time == 0


def test_paragraph_abs_time():
    text = """
aaa bbb ccc ddd. aaa bbb ccc ddd.
aaa bbb ccc ddd. aaa bbb ccc ddd.
    """

    para = Paragraph()
    sent = Sentence()
    para.append_sentence(sent)
    for w, time, duration in split_text_generator(text, start_time=10):
        if sent.is_ended:
            sent = Sentence()
            para.append_sentence(sent)
        word = Word(w, time, duration)
        sent.append(word)
        

    # assert para.time == 10
    assert len(para.sentences) == 4
    assert para[0].time == 10
    assert para[0].duration == 16
    assert para[1].time == 10 + 16 + 1
    assert para[1].duration == 16
    assert para[2].time == 10 + 2 *(16 + 1)
    assert para[2].duration == 16
    assert para[3].time == 10 + 3 *(16 + 1)
    assert para[3].duration == 16

    
    


def test_appending_transcript_abs_time():
    text = """
aaa bbb ccc ddd. aaa bbb ccc ddd.

aaa bbb ccc ddd. aaa bbb ccc ddd. aaa bbb ccc ddd.

aaa bbb ccc ddd. aaa bbb ccc ddd. aaa bbb ccc ddd. aaa bbb ccc ddd.
"""    

    transcript = Transcript() 
    
    for w, time, duration in split_text_generator(text):
        word = Word(w, time, duration)
        transcript.append_word(word)

    assert len(transcript.paragraphs) == 3
    assert len(transcript[0].sentences) == 2
    assert len(transcript[1].sentences) == 3
    assert len(transcript[2].sentences) == 4

    assert transcript[0][0].time == 0
    assert transcript[0][0].duration == 16
    assert transcript[0][1].time == 17
    assert transcript[0][1].duration == 16
    assert transcript[1][0].time == 34
    assert transcript[1][0].duration == 16
    assert transcript[2][0].time == 85
    assert transcript[2][0].duration == 16

    


# 'Ever wondered why cats have a mesmerizing power to melt hearts and effortlessly rule the internet?'
def test_simple_sentence_segments():

    with open('__tests__/data/simple_content.json', 'r') as f:
        lexical_data = json.load(f)
        transcript = Transcript.from_lexical(lexical_data)

    sentence = transcript[0][0]

    segments = sentence.generate_segments()
    
    assert len(segments) == 4
    assert sentence.time <= segments[0].time
    assert sentence.time + sentence.duration >= segments[-1].time + segments[-1].duration
    for i in range(len(segments)-1):
        assert segments[i].time + segments[i].duration <= segments[i+1].time
        assert segments[i].time + segments[i].duration >= segments[i+1].time - 1


    
def test_random_sentence_segments():

    with open('__tests__/data/rabbit_script.json', 'r') as f:
        lexical_data = json.load(f)
        transcript = Transcript.from_lexical(lexical_data)

    sentence_list = [        
        (transcript[0][0], 3),
        (transcript[0][1], 1),
    ]
    for sentence, seg_num in sentence_list:
        segments = sentence.generate_segments()
        
        assert len(segments) == seg_num
        assert 0 <= segments[0].time
        assert sentence.duration >= segments[-1].time + segments[-1].duration
        for i in range(len(segments)-1):
            assert segments[i].time + segments[i].duration - segments[0].time <= segments[i+1].time
            assert segments[i].time + segments[i].duration - segments[0].time >= segments[i+1].time - 1
        
