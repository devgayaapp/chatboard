from pyannote.audio import Pipeline
from config import HUGGING_FACE_TOKEN
from pydub import AudioSegment


def get_diarization_pipeline():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HUGGING_FACE_TOKEN)
    return pipeline


def get_speaker_diarization(audio_file, min_speakers=2, max_speakers=5):
    wav_audio_file = audio_file
    if not wav_audio_file.endswith('.wav'):
        audio = AudioSegment.from_file(audio_file)
        wav_audio_file = wav_audio_file.split('.')[0] + '.wav'
        audio.export(wav_audio_file, format="wav")
    
    pipeline = get_diarization_pipeline()
    diarization = pipeline(wav_audio_file, min_speakers=min_speakers, max_speakers=max_speakers)
    return diarization


def diarization_to_json(diarization):
    speaker_diarization = []
    for seg, track, speaker in diarization.itertracks(yield_label=True):
        speaker_diarization.append({
            'start': seg.start,
            'end': seg.end,
            'duration': seg.duration,
            'speaker': speaker,
            'track': track
        })