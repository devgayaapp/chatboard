import re
import pandas as pd
from html import unescape
from xml.etree import ElementTree
# from components.audio.captions import Captions
from time import time
import hashlib
from pyannote.audio import Audio
from pyannote.core.annotation import Annotation, Segment
from config import DEBUG
# from components.text.punctuation import add_punctuation, merge_punctuation
from components.text.nlp.punctuation import add_punctuation, merge_punctuation

if DEBUG:
    try:
        from IPython import display
    except:
        pass

def display_audio(sub_clip):
    display.display(display.Audio(sub_clip[0], rate=sub_clip[1]))
    
# def display_segment(audio_filepath_wav, segment):
#     subclip = audio.crop(audio_filepath_wav, segment)
#     display_audio(subclip)

# def crop_audio(segment):
#     return audio.crop(yt.audio_filepath_wav, segment)



RE = re.compile(r' ?- ')
TRANSCRIPT_EVENTS = re.compile(r'[\(|\[](.+)[\)|\]]')
SPEAKER_SPLIT = re.compile(r'(?<=-)[^-]*')
REMOVE_ACTION = re.compile(r'\(.+\)')
REMOVE_STARTING_SPACE = re.compile(r'^ ')


class YTWord:

    def __init__(self, text, punct_text, start_time, duration, end_time) -> None:
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.punct_text = punct_text


    def __str__(self):
        return f'[{self.start_time}-{self.end_time}, {self.duration}] ' + self.punct_text

    def __repr__(self):
        return str(self)



class YTWordSpan:

    def __init__(self, words_df, audio_filepath_wav) -> None:
        self.words_df = words_df
        self.words = []
        self.speaker = words_df.iloc[0]['speaker']
        self.track = words_df.iloc[0]['track']
        self.audio_filepath_wav = audio_filepath_wav
        self.pyannote_audio = Audio()
        for i, row in self.words_df.iterrows():
            word = YTWord(row['word'], row['punct_word'], row['start_time'], row['duration'], row['end_time'])
            self.words.append(word)
            
    def __str__(self):
        start_time = self.words[0].start_time
        end_time = self.words[-1].end_time
        return f'[{self.speaker}, {self.track}> {start_time}-{end_time}] ' + ' '.join([word.punct_text for word in self.words])

    def __repr__(self):
        return str(self)

    # def _repr_png_(self):
    #     self.audio()
    #     return str(self)

    def _repr_html_(self):
        self.audio()
        start_time = self.words[0].start_time
        end_time = self.words[-1].end_time

        # return f'[{self.speaker}, {self.track}> {start_time}-{end_time}] ' + 
        sentence = ' '.join([word.punct_text for word in self.words])
        return f'''
<div class="yt-word">
    <div class="yt-speaker" style="color: red;">{self.speaker}</div>
    <div class="yt-word-text">{sentence}</div>
    <span class="yt-word-time" style="color: white; background: blue; border-radius: 5px; padding: 0px 10px 0px 10px;">{start_time}-{end_time}</span>
</div>
'''

    # def _repr_html_(self):
    #     self.audio()
    #     return f'''<div>{str(self)}'''

    def __iter__(self):
        self._curridx = 0
        return self

    def __next__(self):
        if self._curridx < len(self.words):
            word = self.words[self._curridx]
            self._curridx += 1
            return word
        else:
            raise StopIteration()

    def __len__(self):
        return len(self.words)

    def __getitem__(self, key):
        return self.words[key]

    def audio(self):
        return display_audio(self.pyannote_audio.crop(self.audio_filepath_wav, Segment(self.words[0].start_time, self.words[-1].end_time)))


class YTWordSpanList:

    def __init__(self, spans, audio_filepath_wav) -> None:
        spans.sort(key=lambda span: span.words[0].start_time)
        self.spans = spans
        self.audio_filepath_wav = audio_filepath_wav

    def display_html(self):
        for span in self.spans:
            display.display(display.HTML(span._repr_html_()))

    def _repr_html_(self):
        self.display_html()
        return ''

    def __iter__(self):
        self._curridx = 0
        return self

    def __next__(self):
        if self._curridx < len(self.spans):
            span = self.spans[self._curridx]
            self._curridx += 1
            return span
        else:
            raise StopIteration()
    
    def __len__(self):
        return len(self.spans)

    def __getitem__(self, key):
        return self.spans[key]


        

class YTCaptions:

    def __init__(self, caption_obj, diarization, audio_filepath_wav, start_at, end_at):
        self.name = caption_obj.name
        self.code = caption_obj.code
        self.diarization = diarization
        self.start_at = start_at
        self.end_at = end_at
        self.audio_filepath_wav = audio_filepath_wav
        if '(auto-generated)' in caption_obj.name:
            print('auto generated captions')
            captions_json = xml_auto_caption_to_json(caption_obj.xml_captions)
            processed_captions_json = process_auto_caption_json(captions_json, self.start_at, self.end_at)
            add_punctuation = True
        else:
            #! TODO: this code is buggy. the xml_caption_to_json gives diffrent results than the auto caption
            print('no auto captions')
            processed_captions_json = xml_caption_to_json(caption_obj.xml_captions)
            add_punctuation = False
            
        self.captions_df = auto_captions_to_pd(processed_captions_json)
        if len(self.captions_df) > 0:
            self.merge_diarization_captions(add_punctuation=add_punctuation)
        self.curr_word = None


    def merge_diarization_captions(self, add_punctuation=True):
        if self.diarization:
            for seg, track, speaker in self.diarization.itertracks(yield_label=True):
                idxs = self.get_words_idx(seg.start, seg.end)
                self.captions_df.loc[idxs, 'speaker'] = speaker
                self.captions_df.loc[idxs, 'track'] = track
        # for g, df in self.captions_df.groupby(['track']):
        #     last_index = df.index[-1]
        #     self.captions_df.loc[last_index, 'punct_word'] = self.captions_df.loc[last_index, 'punct_word'] + '.'
        full_text = self.captions_df['punct_word'].str.cat(sep=' ')
        print('adding punctuation')
        if add_punctuation:
            punct_text = add_punctuation(full_text, int(len(self.captions_df) * 1.333) + 20)
        else:
            punct_text = full_text
        print('punctuation added')
        # self.tokens = j['completions'][0]['data']['tokens']
        # self.punct_response = j
        self.punct_text = punct_text
        words = self.captions_df[['punct_word', 'word']].to_dict(orient='records')
        punct_words = merge_punctuation(words, punct_text)
        self.captions_df['punct_word'] = [rec['punct_word'] for rec in punct_words]
        period_idx = (self.captions_df['punct_word'].str.endswith('. ')) | (self.captions_df['punct_word'].str.endswith('.'))
        self.captions_df.loc[period_idx, 'has_period'] = True
        for i, idx in enumerate([i for i,v in enumerate(period_idx) if v == True]):
            self.captions_df.loc[idx+1:, 'sentence_num'] = i + 2
        self.captions_df['speaker'] = self.captions_df['speaker'].fillna(method='ffill')
        self.captions_df['track'] = self.captions_df['track'].fillna(method='ffill')
        self.captions_df['speaker'] = self.captions_df['speaker'].fillna(method='bfill')
        self.captions_df['track'] = self.captions_df['track'].fillna(method='bfill')
        
        #? fill NA values with next speaker.
        

        
                

    def get_words_idx(self, start_time: float, end_time: float):
        return (self.captions_df['start_time'] >= start_time) & (self.captions_df['end_time'] <= end_time)

    def get_words(self, start_time: float, end_time: float):
        words_df = self.captions_df[(self.captions_df['start_time'] >= start_time) & (self.captions_df['end_time'] <= end_time)]
        word_spans = []
        for i, track_df in words_df.groupby(['track']):
            word_spans.append(YTWordSpan(track_df, self.audio_filepath_wav))
        return YTWordSpanList(word_spans, self.audio_filepath_wav)
        # words_df.to_dict(orient='records')
        return YTWordSpan(words_df, self.audio_filepath_wav)

    def _repr_png_(self):
        word_span_list = self.get_words(self.start_at, self.end_at)
        word_span_list.display_html()
        return self.diarization._repr_png_()

    def get_speakers(self):
        return self.diarization.labels()

    def get_speaker_captions(self, speaker):
        return self.captions_df[self.captions_df['speaker'] == speaker]

    def get_timeline(self):
        return self.diarization.get_timeline()

    def get_sentences_df(self):
        group_df = self.captions_df.groupby(['sentence_num'])
        df = group_df['punct_word'].apply(lambda x: ' '.join(x)).reset_index()
        df['start_time'] = group_df['start_time'].min()
        df['end_time'] = group_df['end_time'].max()
        df['duration'] = df['end_time'] - df['start_time']
        df['speaker'] = group_df['speaker'].first()
        df['track'] = group_df['track'].first()
        return df



    def to_json(self):
        # self.captions_df.to_dict(orient='records')
        transcript_json = []        
        curr_sentence_num = 1
        words = []
        speaker = None
        track = None
        for i, row in self.captions_df.iterrows():
            if row['sentence_num'] != curr_sentence_num:
                transcript_json.append({
                    'words': words,
                    'start': words[0]['time'],
                    'end': words[-1]['time'] + words[-1]['duration'],
                    'duration': words[-1]['time'] + words[-1]['duration'] - words[0]['time'],
                    'speaker': speaker,
                    'track': track,
                })
                words = []
                curr_sentence_num = row['sentence_num']
            speaker = row['speaker']
            track = row['track']
            words.append({
                'word': row['word'],
                'punct_word': row['punct_word'],
                'time': row['start_time'],
                'duration': row['duration'],
                'has_period': row['has_period'],
            })
        return {
            'lang': self.code,
            'transcript': transcript_json,
        }

    

    



def get_full_text(processed_transcript):
    words = []
    for sentence in processed_transcript:
        for w in sentence['words']:
            words.append(w['word'])
    full_text = ' '.join(words)
    return full_text





def auto_captions_to_pd(processed_captions_json):
    words = []
    for i, sentence in enumerate(processed_captions_json):
        for word in sentence['words']:
            words.append({
                'word': word['word'],
                'punct_word': word['word'],
                'start_time': word['time'],
                'duration': word['duration'],
                'end_time': word['time'] + word['duration'],
                'speaker': None,
                'track': None,
                'has_period': False,
                'sentence_num': 1,

            })
    df = pd.DataFrame(
        words, 
        columns=['word', 'punct_word', 'start_time', 'duration', 'end_time', 'speaker', 'track', 'has_period', 'sentence_num'],        
        )
    # df['word'] = df['word'].astype(str)
    # df['punct_word'] = df['punct_word'].astype(str)    
    return df

# word             object
# punct_word       object
# start_time      float64
# duration        float64
# end_time        float64
# speaker          object
# track            object
# has_period         bool
# sentence_num      int64

def xml_caption_to_json(xml_captions, camption_obj=None):
    """Convert xml caption tracks to "SubRip Subtitle (srt)".

    :param str xml_captions:
    XML formatted caption tracks.
    """
    segments = []
    segment_objs = []
    root = ElementTree.fromstring(xml_captions)
    i=0
    for child in list(root.iter("body"))[0]:
        if child.tag == 'p':
            caption = ''
            if len(list(child))==0:
                # instead of 'continue'
                caption = child.text
            for s in list(child):
                if s.tag == 's':
                    caption += ' ' + s.text
            caption = unescape(caption.replace("\n", " ").replace("  ", " "),)
            caption = REMOVE_STARTING_SPACE.sub('', caption)
            # caption_na = REMOVE_ACTION.sub('', caption).replace(r"  +", ' ');
            events = TRANSCRIPT_EVENTS.findall(caption)
            speakers = []
            for group in SPEAKER_SPLIT.finditer(caption):
                pos = group.regs[0]
                speakers.append({
                    'text': group.string[pos[0]:pos[1]],
                    'pos': pos,
                    'isStart': pos[0] == 0 or group.string[pos[0]-1] == '-'
                })

            try:
                duration = float(child.attrib["d"])/1000.0
            except KeyError:
                duration = 0.0
            start = float(child.attrib["t"])/1000.0
            end = start + duration
            sequence_number = i + 1  # convert from 0-indexed to 1.

            words = []
            char_duration = duration / len(caption)
            for i, word in enumerate(caption.split(' ')):
                word_duration = char_duration * len(word)
                words.append({
                    'word': word,
                    'punct_word': word,
                    'time': start + i * char_duration,
                    'duration': word_duration,
                    'speaker': None,
                    'track': None,
                    'has_period': False,
                    'sentence_num': 1,
                })

            line_obj = {
                'sentence': caption,
                'start': start,
                'duration': duration,
                'end': end,
                'events': events,
                'speakers': speakers,
                "words": words
            }
            if camption_obj:
                line = "{seq}\n{start} --> {end}\n{text}\n".format(
                    seq=sequence_number,
                    start=camption_obj.float_to_srt_time_format(start),
                    end=camption_obj.float_to_srt_time_format(end),
                    text=caption,
                )
                segments.append(line)
            segment_objs.append(line_obj)
            i += 1
    if camption_obj:
        return segment_objs, "\n".join(segments).strip()
    return segment_objs




#https://jacobstar.medium.com/the-first-complete-guide-to-youtube-captions-f886e06f7d9d
def xml_auto_caption_to_json(xml_captions):
    """Convert xml caption tracks to "SubRip Subtitle (srt)".

    :param str xml_captions:
    XML formatted caption tracks.
    """
    segments = []
    segment_objs = []
    root = ElementTree.fromstring(xml_captions)
    i=0
    all_sentence_list = []
    root_body_list = list(root.iter("body"))[0]
    for sentence_idx, sentence in enumerate(root_body_list):

        if sentence.tag == 'p':
            caption = ''
            is_wait = False            
            # if len(list(sentence))==0:
            if len(list(sentence))==0:
                try:
                    is_wait = (sentence.attrib['a'] == "1")
                except KeyError:                    
                    continue

            word_json_list = []
            sentence_list = list(sentence)
            for w_i, word in enumerate(sentence_list):
                if word.tag == 's':
                    caption += ' ' + word.text
                    t = word.attrib.get("t", None)
                    if t is None:                          
                        word_json_list.append({
                            "w": word.text, 
                            "rel_time": None
                        })
                        continue
                    word_duration = float(t)/1000.0
                    if word_duration > 15.0:
                        raise Exception("Word duration is too long")
                    word_json_list.append({
                        "w": word.text, 
                        "rel_time": word_duration 
                    })
            #fix all missing values with interpolation
            if len(word_json_list) == 0:
                continue
            sentence_duration = int(sentence.attrib['d'])/1000.0
            mean_word_duration = sentence_duration / len(word_json_list)
            for i, w in enumerate(word_json_list):
                if w['rel_time'] == None:
                    if i == 0:
                        word_json_list[i]['rel_time'] = 0.0
                    elif i == len(word_json_list)-1:
                        if len(word_json_list) >= 3:
                            # word_json_list[i]['rel_time'] = word_json_list[-2]['rel_time'] - word_json_list[-3]['rel_time']
                            word_json_list[i]['rel_time'] = word_json_list[-1]['rel_time'] - word_json_list[-2]['rel_time']
                        else:
                            word_json_list[i]['rel_time'] = mean_word_duration
                    else:
                        if word_json_list[i+1]['rel_time'] is not None:
                            word_json_list[i]['rel_time'] = (word_json_list[i-1]['rel_time'] + word_json_list[i+1]['rel_time'])/2.0
                        else:
                            word_json_list[i]['rel_time'] = word_json_list[i-1]['rel_time'] + mean_word_duration

            

            try:
                duration = float(sentence.attrib["d"])/1000.0
            except KeyError:
                if is_wait:
                    duration = 0.0
                else:
                    raise Exception("No duration for sentence")
            start = float(sentence.attrib["t"])/1000.0

            all_sentence_list.append({
                'time': start,
                'dur': duration,
                'words': word_json_list,
                'wait': is_wait
            })

            i += 1        
    return all_sentence_list



# def process_auto_caption_json(auto_caption_json):
    
#     sentence_transcript = []
#     for i, sentence in enumerate(auto_caption_json):
#         word_transcript = []
#         if sentence['wait']:
#             continue
#         sentence_time = sentence['time']
#         sentence_duration = auto_caption_json[i+1]['time'] - auto_caption_json[i]['time'] if i != len(auto_caption_json)-1 else auto_caption_json[i]['dur']
#             # sentence_time, sentence_duration = clip_time_duration(sentence_time, sentence_duration)
#         for iw, word in enumerate(sentence['words']):
#             word_duration = sentence['words'][iw+1]['rel_time'] - word['rel_time'] if iw != len(sentence['words'])-1 else sentence_duration - word['rel_time']                
#             word_time = sentence_time + word['rel_time']                
#             word_transcript.append({
#                 'word': word['w'][1:] if word['w'].startswith(' ') else word['w'],
#                 'duration': round(word_duration, 3),
#                 'time': round(word_time, 3),
#             })
#         # sentence_time, sentence_duration = clip_time_duration(sentence_time, sentence_duration)
#         sentence = ' '.join([word['w'] for word in sentence['words']])
#         sentence_transcript.append({
#             'hash':  hashlib.md5((sentence + str(time())).encode('utf-8')).hexdigest(),
#             'sentence': sentence,
#             'duration': round(sentence_duration, 3),
#             'time': round(sentence_time, 3),
#             'words': word_transcript
#         })           
#     return sentence_transcript


def process_auto_caption_json(auto_caption_json, start_at, end_at):

    def clip_time_duration(time, duration):
        cliped_time = min(max(time, start_at), end_at)
        if cliped_time + duration > end_at:
            duration = end_at - time
        if time < start_at:
            duration = duration - (start_at - time)
        return cliped_time, duration
        
    sentence_transcript = []
    for i, sentence in enumerate(auto_caption_json):
        word_transcript = []
        if sentence['wait']:
            continue
        sentence_time = sentence['time']
        sentence_duration = auto_caption_json[i+1]['time'] - auto_caption_json[i]['time'] if i != len(auto_caption_json)-1 else auto_caption_json[i]['dur']
        if sentence_time + sentence_duration >= start_at and sentence_time <= end_at:
            # sentence_time, sentence_duration = clip_time_duration(sentence_time, sentence_duration)
            for iw, word in enumerate(sentence['words']):
                word_duration = sentence['words'][iw+1]['rel_time'] - word['rel_time'] if iw != len(sentence['words'])-1 else sentence_duration - word['rel_time']                
                word_time = sentence_time + word['rel_time']                
                if word_time >= start_at and word_time <= end_at:
                    word_time, word_duration = clip_time_duration(word_time, word_duration)
                    word_transcript.append({
                        'word': word['w'][1:] if word['w'].startswith(' ') else word['w'],
                        'duration': round(word_duration, 3),
                        'time': round(word_time - start_at, 3),
                    })
            sentence_time, sentence_duration = clip_time_duration(sentence_time, sentence_duration)
            sentence = ' '.join([word['w'] for word in sentence['words']])
            sentence_transcript.append({
                'hash':  hashlib.md5((sentence + str(time())).encode('utf-8')).hexdigest(),
                'sentence': sentence,
                'duration': round(sentence_duration, 3),
                'time': round(sentence_time - start_at, 3),
                'words': word_transcript
            })           
    return sentence_transcript
