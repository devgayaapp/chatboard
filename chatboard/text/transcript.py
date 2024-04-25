import json
import re
from typing import Union
from uuid import uuid4



class TranscriptError(Exception):
    pass


def sent2num(sentiment):
    if sentiment == 'Positive':
        return 1
    elif sentiment == 'Negative':
        return -1
    else:
        return 0
    

def num2sent(sentiment):
    if sentiment >= 1:
        return 'Positive'
    elif sentiment <= -1:
        return 'Negative'
    else:
        return 'Neutral'







class Word:
    """
    Container for a word in a transcript

    Attributes
    ----------
    time: float
        time of the word in a sentence. relative to the sentence time.
    duration: float
        duration of the word
    """

    def __init__(self, text, time=0, duration=1, score=0, sentiment='neutral', entity=None, isProcessed=False, phrase_index='A', color=None, background_color=None) -> None:
        self.text = text
        self.time = time
        self.duration = duration
        self.score = score
        self.sentiment = sentiment
        self.isProcessed = isProcessed
        self.phrase_index = phrase_index
        self.entity = entity
        self.color = color
        self.background_color = background_color
        self.sentence_ref = None

    @property
    def segment(self):
        return self.phrase_index

    @property
    def end_time(self):
        return self.time + self.duration

    def to_lexical(self):
        return {
            'type': 'word',
            'text': self.text,
            'time': self.time,
            'duration': self.duration,
            'score': self.score,
            'sentiment': self.sentiment,
            # 'isProcessed': self.isProcessed,
            'phraseIndex': self.phrase_index,
            "color": self.color,
            "backgroundColor": self.background_color,
            "entity": self.entity,
            "version": 1,
            "detail": 0,
            "mode": "normal",
            "style": "",
            "format": 0,
        }
    
    def to_json(self):
        return {
            'type': 'word',
            'text': self.text,
            'time': self.time,
            'duration': self.duration,
            'score': self.score,
            'sentiment': self.sentiment,
            # 'isProcessed': self.isProcessed,
            'phraseIndex': self.phrase_index,
            "color": self.color,
            "backgroundColor": self.background_color,
            "entity": self.entity,            
        }
    
    def to_string(self):
        return f'{self.text} {self.duration:.2f}-[{self.time:.2f}, {self.time + self.duration:.2f}]  {self.phrase_index} Snt:{self.sentiment} Scr:{self.score:.2f}'
    
    def __repr__(self) -> str:
        return f"Word: {self.to_string()}"
    
    def __str__(self) -> str:
        return self.to_string()
    
    # def __repr__(self) -> str:
    #     return self.to_string()

    @property
    def abs_time(self):
        if self.sentence_ref:
            return self.sentence_ref.time + self.time
        return self.time
    
    @property
    def is_end_of_sentence(self):
        return len(re.findall(r'[.!?]\s*$', self.text)) > 0 or self.is_end_of_paragraph
    
    @property
    def is_punctuation(self):
        return len(re.findall(r'[.!?]', self.text)) > 0
    
    @property
    def is_ends_with_space(self):
        return len(re.findall(r'\s*$', self.text)) > 0
    
    @property
    def is_end_of_paragraph(self):
        return len(re.findall(f'\n+\s*$', self.text)) > 0
    
        


class Sentence:
    """
    Container for Transcript Sentence

    Attributes
    ----------
    time: float
        time when the sentence is starting. relative to the screen time
    duration: float
        the duration of the sentence
    """

    def __init__(self, text=None, time=None, duration=None, words=None, uuid=None, speaker=None, track=None, is_voice_generated=False, audio=None, audioText=None, is_dirty=True, is_ai_generated=True, metadata=None) -> None:

        
        if text is not None and words is None:
            words = []
            for w in [w for w in text.split(' ') if w != '']:
                word = Word(w+' ')
                if len(words) and words[-1].is_end_of_sentence:
                    print("Warning: Text contains multiple sentences")
                    # raise TranscriptError('Text contains multiple sentences')
                words.append(word)

        self._time = time
        self._duration = duration
        self.words = words if words is not None else []        
        self.set_word_sentence_ref()
        self.speaker = speaker
        self.uuid = uuid or str(uuid4())[0:6]
        self.track = track
        self.is_dirty = is_dirty
        self.audio = audio
        # if audioText is None:
        #     self.generate_audio_text()
        # else:
        #     self.audioText = audioText
        self.audioText = audioText
        self.is_voice_generated = is_voice_generated
        self.is_ai_generated = is_ai_generated
        self.metadata = metadata or {}

    def set_word_sentence_ref(self):
        for w in self.words:
            w.sentence_ref = self

    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self.words):
            return None
        return self.words[index]
    
    def add_metadata(self, key, value):
        self.metadata[key] = value
    
    @property
    def time(self):
        if self._time is not None:
            return self._time
        if len(self.words) > 0:
            return self.words[0].time
        
    @property
    def duration(self):
        if self._duration is None:
            return self.wordDuration
        return self._duration
    
    @property
    def wordDuration(self):
        if len(self.words) > 0:
            return self.words[-1].time + self.words[-1].duration - self.words[0].time
        return 0

    @property
    def text(self):
        return ''.join([w.text for w in self.words])
    

    @property
    def is_end_of_paragraph(self):
        return self.words[-1].is_end_of_paragraph if self.words else False
    

    @property
    def last_word(self):
        return self.words[-1] if self.words else None
    
    @property
    def is_ended(self):
        if self.last_word:
            return self.last_word.is_end_of_sentence

    def append(self, word: Word, fix_abs_time=True):
        if len(self.words) == 0:
            self._time = word.time
        if fix_abs_time:
            word.time -= self.time
        word.sentence_ref = self
        self.words.append(word)
        # self.generate_audio_text()


    def generate_audio_text(self):
        self.audioText = ''.join([w.text for w in self.words])


    def setTime(self, time, fix_abs_time=False):
        if fix_abs_time:
            for w in self.words:
                w.time -= time
        self._time = time

    def to_json(self):
        return {        
            'time': self.time,
            'duration': self.duration,
            'words': [w.to_json() for w in self.words],
            'speaker': self.speaker,
            'uuid': self.uuid,
        }
    
    def to_lexical(self):
        return {
            'type': 'sentence',            
            'children': [w.to_lexical() for w in self.words],
            'time': self.time,
            'duration': self.duration,            
            # 'speaker': self.speaker,
            'uuid': str(self.uuid),
            'audio': self.audio,
            'audioText': self.audioText,
            # 'isVoiceGenerated': self.is_voice_generated,
            'isAiGenerated': self.is_ai_generated,
            'audio': self.audio,
            'audioText': self.audioText,
            'isDirty': self.is_dirty,
            "indent": 0,
            "direction": None,
            "version": 1,
            'metadata': self.metadata,
            
        }
    
    @staticmethod
    def from_lexical(lex_sentence):
        words = []
        for lex_word in lex_sentence['children']:
            word = Word(
                text=lex_word['text'],
                time=lex_word['time'],
                duration=lex_word['duration'],
                score=lex_word.get('score', 0),
                sentiment=lex_word.get('sentiment', 'neutral'),
                isProcessed=lex_word.get('isProcessed',False),
                phrase_index=lex_word.get('phraseIndex', 'A'),
                color=lex_word.get('color', None),
                background_color=lex_word.get('backgroundColor', None),
            )
            words.append(word)
        sentence = Sentence(
            time=lex_sentence['time'],
            duration=lex_sentence['duration'],
            words=words,
            speaker=lex_sentence.get('speaker', 0),
            uuid=lex_sentence['uuid'],
            is_dirty=lex_sentence['isDirty'],
            audio=lex_sentence.get('audio', None),
            audioText=lex_sentence.get('audioText', None),
            is_ai_generated=lex_sentence['isAiGenerated'],
            metadata=lex_sentence.get('metadata', None),
        )
        return sentence

    def to_dict(self):
        return self.to_lexical()
    
    def to_string(self):
        return f"{''.join([w.text for w in self.words])} {self.duration:.2f} [{self.time:.2f}, {self.time + self.duration:.2f}]"
    
    def __str__(self) -> str:
        return self.to_string()
    

    def __repr__(self) -> str:
        return "Sentence(" + self.to_string() + "\n)"
        
    

    
    def generate_segments(self, is_sort=False):
        """the segments are generated relative to the sentence time not absolute time"""
        if len(self.words) == 0:
            return []
        # curr_phrase_idx = self.words[0].phrase_index
        groups = []
        curr_group = Segment(self.words[0])
        for w in self.words:
            if curr_group.is_part_of_group(w):
                curr_group.add_word(w)
            else:
                groups.append(curr_group)
                curr_group = Segment(w)
        groups.append(curr_group)
        if is_sort:
            groups = sorted(groups, key=lambda x: x.score, reverse=True)   
        return groups



    
    # def __repr__(self) -> str:
    #     return self.to_string()
    
    


class Paragraph:
    
    def __init__(self, sentences=None, speaker=None, track=None, is_ai_generated=True, metadata=None, time=None) -> None:
        self.sentences = sentences if sentences is not None else []
        self._curr_sentence = None
        self.speaker = speaker
        self.track = track
        self.is_ai_generated = is_ai_generated
        self._time = time
        self.metadata = metadata or {}

    def __len__(self):
        return sum([len(s) for s in self.sentences])

    def __getitem__(self, index):
        if index < 0 or index >= len(self.sentences):
            return None
        return self.sentences[index]
    
    def add_metadata(self, key, value):
        self.metadata[key] = value
    
    @property
    def duration(self):
        return sum([s.duration for s in self.sentences]) if len(self.sentences) > 0 else None
    
    @property
    def time(self):
        return self.sentences[0].time if len(self.sentences) > 0 else None
    
    @property
    def last_sentence(self):
        return self.sentences[-1] if len(self.sentences) > 0 else None
    
    @time.setter
    def time(self, value):
        self._time = value

    def to_json(self):
        return {
            'sentences': [s.to_json() for s in self.sentences]
        }
    
    def to_lexical(self):
        return {
             "type": "paragraph",
            "format": "",
            "indent": 0,
            "version": 1,
            'children': [s.to_lexical() for s in self.sentences if s.duration is not None and s.duration > 0],
            "direction": "ltr",
            "metadata": self.metadata,
        }
    
    def to_dict(self):
        return self.to_lexical()
    

    @property
    def text(self):
        return '\n'.join([s.text for s in self.sentences])
    
    def to_string(self):
        return '\n'.join([s.to_string() for s in self.sentences])

    def __str__(self) -> str:
        return self.to_string()
    
    def __repr__(self) -> str:
        return "Paragraph(\n" + self.to_string() + "\n)"
    
    @property
    def curr_sentence(self):
        if not self._curr_sentence:
            self._curr_sentence = self.last_sentence
        return self._curr_sentence
    

    def append_sentence(self, sentence=None, speaker=None, track=None, is_ai_generated=None):
        if sentence is None:
            sentence = Sentence(
                speaker=self.speaker if speaker is None else speaker,
                track=self.track if track is None else track,
                is_ai_generated=self.is_ai_generated if is_ai_generated is None else is_ai_generated
            )
        # start_time = self._curr_sentence.time if self._curr_sentence else self.time
        # sentence._time = start_time
        self.sentences.append(sentence)
        self._curr_sentence = sentence

    @property
    def last_sentence(self):
        return self.sentences[-1] if len(self.sentences) > 0 else None

    @property    
    def is_ended(self):        
        if self.last_sentence and self.last_sentence.last_word:
            return self.last_sentence.last_word.is_end_of_paragraph
        return False
        


class Transcript:

    def __init__(self, lang='en-us', paragraphs=None, is_ai_generated=True) -> None:
        self.lang = lang
        self.paragraphs = paragraphs if paragraphs is not None else []        
        self._curr_paragraph = None
        self.is_ai_generated = is_ai_generated
        self.sentence_dict = None


    def __getitem__(self, index: Union[str, int]):
        if type(index) == str:
            if self.sentence_dict is None:
                self.generate_sentence_ids_dict()
            return self.sentence_dict[index]
        
        if index < 0 or index >= len(self.paragraphs):
            return None
        return self.paragraphs[index]

    def __len__(self):
        return sum([len(p) for p in self.paragraphs])



    @property
    def last_paragraph(self):
        if self.paragraphs:
            return self.paragraphs[-1]
        return None
    

    def generate_sentence_ids_dict(self):
        self.sentence_dict = {}
        for i, p in enumerate(self.paragraphs):
            for j, s in enumerate(p.sentences):
                self.sentence_dict[s.uuid] = s
    


    def append_word(self, word: Word, speaker=None, track=None, fix_abs_time=True):

        if self.curr_paragraph is None or self.curr_paragraph.is_ended or self.curr_paragraph.speaker != speaker or self.curr_paragraph.track != track:
            self.append_paragraph(speaker=speaker, track=track, is_ai_generated=self.is_ai_generated)
        if self.curr_sentence is None or self.curr_sentence.is_ended:
            prev_sentence = self.curr_sentence
            self.curr_paragraph.append_sentence(speaker=speaker, track=track, is_ai_generated=self.is_ai_generated)

        self.curr_sentence.append(word)

        # if speaker is not None:
        #     if self.curr_sentence.speaker is None:
        #         self.curr_sentence.speaker = speaker
        #     elif speaker != self.curr_sentence.speaker:
        #         sentence = Sentence(speaker=speaker, track=track, is_ai_generated=self.is_ai_generated)
        #         self._curr_paragraph.append_sentence(sentence)
        # #TODO fix ABS time
        # if fix_abs_time: 
        #     if self.curr_sentence.time is None:
        #         self.curr_sentence._time = word.time
        #     word.time -= self.curr_sentence.time
        
        # if word.is_end_of_sentence:
        #     sentence = Sentence(is_ai_generated=self.is_ai_generated)
            

        # if word.is_end_of_paragraph:
        #     self.append_paragraph()


    def unique_speakers(self):
        return set([s.speaker for p in self.paragraphs for s in p.sentences if s.speaker is not None])


    def to_json(self):
        return {
            'lang': self.lang,
            'paragraphs': [p.to_json() for p in self.paragraphs],
        }


    def to_lexical(self):
        return {
            'root': {
                "type": "root",
                "format": "",
                "indent": 0,
                "version": 1,
                'children': [pl for pl in [p.to_lexical() for p in self.paragraphs] if len(pl['children']) > 0],
                "direction": None
            }
        }
    
    
    @property
    def curr_paragraph(self):
        if self._curr_paragraph is None:
            self._curr_paragraph = self.last_paragraph

        return self._curr_paragraph            
    

    def append_paragraph(self, paragraph=None, speaker=None, track=None, is_ai_generated=None):
        if paragraph is None:
            last_speaker = self.last_paragraph.speaker if self.last_paragraph else None
            last_track = self.last_paragraph.track if self.last_paragraph else None
            paragraph = Paragraph(
                speaker=speaker if speaker is not None else last_speaker,
                track=track if track is not None else last_track,
                is_ai_generated=self.is_ai_generated if is_ai_generated is None else is_ai_generated
            )
        if paragraph.curr_sentence is None:
            paragraph.append_sentence(speaker=speaker, track=track, is_ai_generated=self.is_ai_generated)

        start_time = self._curr_paragraph.time if self._curr_paragraph else 0

        self.paragraphs.append(paragraph)

        paragraph.time = start_time

        self._curr_paragraph = paragraph

        # if paragraph is not None:
        #     self._curr_paragraph = paragraph
        #     self.paragraphs.append(paragraph)
        # else:
        #     self._curr_paragraph = Paragraph(speaker=speaker, track=track, is_ai_generated=self.is_ai_generated)
        #     self.paragraphs.append(self._curr_paragraph)


    # def append_sentence(self, sentence: Sentence):
    #     self.curr_paragraph.append_sentence(sentence, is_ai_generated=self.is_ai_generated)        
    #     if sentence.is_end_of_paragraph:
    #         self.append_paragraph()



    @property
    def curr_sentence(self):
        return self.curr_paragraph.curr_sentence
    

    @staticmethod
    def from_lexical(content, is_ai_generated=True):
        paragraphs = []
        for i, lex_paragraph in enumerate(content['root']['children']):
            sentences = []
            speaker = None
            track = None
            for lex_sentence in lex_paragraph['children']:
                words = []
                for lex_word in lex_sentence['children']:
                    word = Word(
                        text=lex_word['text'],
                        time=lex_word['time'],
                        duration=lex_word['duration'],
                        score=lex_word.get('score', 0),
                        sentiment=lex_word.get('sentiment', 'neutral'),
                        isProcessed=lex_word.get('isProcessed',False),
                        phrase_index=lex_word.get('phraseIndex', 'A'),
                        color=lex_word.get('color', None),
                        background_color=lex_word.get('backgroundColor', None),
                    )
                    words.append(word)
                speaker = lex_sentence.get('speaker', speaker)
                track = lex_sentence.get('track', track)
                sentence = Sentence(
                    time=lex_sentence['time'],
                    duration=lex_sentence['duration'],
                    words=words,
                    speaker=lex_sentence.get('speaker', 0),
                    uuid=lex_sentence['uuid'],
                    is_dirty=lex_sentence['isDirty'],
                    audio=lex_sentence.get('audio', None),
                    audioText=lex_sentence.get('audioText', None),
                    is_ai_generated=lex_sentence['isAiGenerated'],
                    metadata=lex_sentence.get('metadata', None),
                )
                sentences.append(sentence)
            paragraph = Paragraph(
                sentences=sentences,
                is_ai_generated=is_ai_generated,
                speaker=speaker,
                track=track,
                metadata=lex_paragraph.get('metadata', None),
            )
            paragraphs.append(paragraph)            
        
        transcript = Transcript(paragraphs=paragraphs)
        transcript.generate_sentence_ids_dict()
        return transcript
    
    
    
    def replace_sentences(self, new_sentences):
        new_sentence_dict = {s.uuid: s for s in new_sentences}
        current_time = 0
        for p in self.paragraphs:
            for i,s in enumerate(p.sentences):                
                if s.uuid in new_sentence_dict:
                    p.sentences[i] = new_sentence_dict[s.uuid]
                p.sentences[i].setTime(current_time)
                current_time += p.sentences[i].duration

    @staticmethod
    def from_json(transcript_json):
        transcript = Transcript()
        for sent in transcript_json:
            for w in sent['words']:
                transcript.append_word(Word(
                        text=w['punct_word'],
                        time=w['time'],
                        duration=w['duration'],
                        score=w['score'],
                        sentiment=w['sentiment'],
                        entity=w['entity_label'],
                        phrase_index=w['group']
                    ))

        return transcript

    def to_string(self):
        return '\n\n'.join([p.to_string() for p in self.paragraphs])

    def to_dict(self):
        return self.to_lexical()
    
    @property
    def text(self):
        return '\n\n'.join([p.text for p in self.paragraphs])


    def get_sentences(self):
        sentences = []
        for p in self.paragraphs:
            for s in p.sentences:
                sentences.append(s)
        return sentences


    @staticmethod
    def from_file(path):
        with open(path, 'r') as f:
            return Transcript.from_lexical(json.load(f))


    def to_file(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_lexical(), f, indent=4)











class Segment:
    """
    Container binds multiple words into a segment with the same context togather.

    Attributes
    ----------
    time: float
        time of the segment
    duration: float
        duration of the segment
    """

    def __init__(self, word: Word) -> None:
        self.words = [word]
        self.id = word.phrase_index
        self.score = word.score
        self.sentiment = word.sentiment
        self.duration = word.duration if word.duration is not None else 0
        # self.time = word.time if word.time is not None else 0
        self.image = None

    @property
    def text(self):
        return ' '.join([w.text for w in self.words])
        
    def add_word(self, word: Word):
        if word.phrase_index != self.id:
            raise Exception(f"Cannot add word {word} to phrase index {self.id}")
        self.words.append(word)
        self.score = sum([w.score for w in self.words]) / len(self.words)
        self.sentiment = num2sent(sum([sent2num(w.sentiment) for w in self.words]))
        if len(self.words) > 1 and self.words[0].time is not None and self.words[-1].time is not None:
            self.duration = self.words[-1].time + self.words[-1].duration - self.words[0].time
            # self.time = self.words[0].time

    @property
    def time(self):
        if self.words:
            return self.words[0].time
        return 0
    
    @property
    def abs_time(self):
        if self.words:
            return self.words[0].abs_time
        return 0

    def is_part_of_group(self, word):
        return word.phrase_index == self.id

    def __str__(self) -> str:
        phrase_str =  f"{self.id}: {' '.join([w.text for w in self.words])} ({self.sentiment}, {round(self.score, 2)})"
        if self.time is not None and self.duration is not None:
            phrase_str += f" {round(self.duration,2)} sec [{round(self.time,2)} - {round(self.time + self.duration, 2)}]"

        if self.image is not None:
            phrase_str += f" *{self.image['image_type']}* - {self.image['description']}"
        return phrase_str
    
    def __repr__(self) -> str:
        return self.__str__()









class SentencePrompt:

    def __init__(self, sentence, prompt, topic=None) -> None:
        self.sentence = sentence
        self.prompt = prompt
        self.topic = topic

    
    @property
    def time(self):
        return self.sentence.time
    
    @property
    def duration(self):
        return self.sentence.duration
    

    