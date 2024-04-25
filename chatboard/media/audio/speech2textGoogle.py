import io
from google.oauth2 import service_account
# from google.cloud import speech
from google.cloud import speech_v1p1beta1 as speech
from components.text.transcript import Transcript, Word
from components.audio.audio import Audio



# client_file = 'api-project-330102785801-8504a831abdc.json'
client_file = 'muspark-395909-263bea589c9b.json'

'''
model types:
latest_long
video
phone_call
command_and_search
'''


def get_encoding(audio_file: Audio):
    if audio_file.ext == 'mp3':
        return speech.RecognitionConfig.AudioEncoding.MP3
    elif audio_file.ext == 'wav':
        return speech.RecognitionConfig.AudioEncoding.LINEAR16
    elif audio_file.ext == 'webm':
        return speech.RecognitionConfig.AudioEncoding.MP3
        # return speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
    else:
        raise ValueError(f'Unrecognized audio file extension: {audio_file.ext}')



def speech2textGoogle(audio_file: Audio, language_code='en-US', model='video'):
    credentials = service_account.Credentials.from_service_account_file(client_file)
    client = speech.SpeechClient(credentials=credentials)
    
    audio = speech.RecognitionAudio(content=audio_file.to_bytes())

    encoding = get_encoding(audio_file)

    config = speech.RecognitionConfig(
        # encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        encoding=encoding,
        sample_rate_hertz=audio_file.sample_rate,
        language_code=language_code,
        # model='latest_long',
        model=model,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        enable_speaker_diarization=True,
        diarization_config=speech.SpeakerDiarizationConfig(
            enable_speaker_diarization=True,
            min_speaker_count=1,
            max_speaker_count=4
        ),
    )
    response = client.recognize(config=config, audio=audio)
    return response



# response.results[0].alternatives[0].words
from string import punctuation

def has_punctuation(word):
    return any([p in word for p in punctuation])

def has_period(word):
    return '.' in word

def strip_punctuation(word):
    return ''.join([c for c in word if not has_punctuation(c)])




def google_speech2text_to_transcript(response):    
    language_code = response.results[0].language_code
    transcript = Transcript(lang=language_code)
    
    for track, result in enumerate(response.results):
        res_transcript = result.alternatives[0].transcript
        res_words = result.alternatives[0].words
        confidence = result.alternatives[0].confidence
        print('confidence:', confidence, 'word num:', len(res_words), '\n', res_transcript, '\n')
        if len(res_words) == 0 or confidence == 0.0 or res_transcript == '':
            continue

        for w in res_words:
            word = Word(
                text= w.word+' ',
                time= w.start_time.total_seconds(),
                duration= round(w.end_time.total_seconds() - w.start_time.total_seconds(),2)
            )            
            transcript.append_word(word, track=track, speaker=res_words[0].speaker_tag)
    return transcript



def google_speech2text_to_transcript2(response):
    transcript_json = []
    current_time = 0
    
    language_code = response.results[0].language_code
    for track, result in enumerate(response.results):
        transcript = result.alternatives[0].transcript
        words = result.alternatives[0].words
        if len(words) == 0:
            print(transcript)
            continue
        transcript_json.append({
            'start': words[0].start_time.total_seconds(),
            'end': round(words[0].start_time.total_seconds() + result.result_end_time.total_seconds(), 2),
            'duration': result.result_end_time.total_seconds(),
            'sentence': transcript,
            'speaker': words[0].speaker_tag,
            'track': track,
            'words': [{
                'word': strip_punctuation(w.word),
                'punct_word': w.word+' ',
                'time': w.start_time.total_seconds(),
                'duration': round(w.end_time.total_seconds() - w.start_time.total_seconds(),2),
                'has_period': has_period(w.word),
            } for i,w in enumerate(words)]
        })
        current_time += result.result_end_time.total_seconds()
    return {
        'lang': language_code,
        'transcript': transcript_json,        
    }
            
