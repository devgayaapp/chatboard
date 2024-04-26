from components.scrapping.mp_pytube import YouTube, Caption
# from components.text.punctuation import add_punctuation, merge_punctuation
from components.text.nlp.punctuation import add_punctuation, merge_punctuation
from components.video.video_shots import VideoShots
from config import DATA_DIR, TEMP_DIR, DEBUG
from pydub import AudioSegment
import json
import os
import ffmpeg
from components.errors import MediaGenerationError
from components.audio.captions import YTCaptions, xml_auto_caption_to_json, process_auto_caption_json, xml_caption_to_json
from components.audio.diatization import get_diarization_pipeline, get_speaker_diarization, diarization_to_json
from pyannote.audio import Audio
import librosa

if DEBUG:
    import IPython.display as ipd




class YoutubeVideo:

    def __init__(self, url, media_name=None, start_at=None, end_at=None):
        self.url = url
        self.yt = YouTube(
            url,
            # use_oauth=True,
            # allow_oauth_cache=True
            )
        if not media_name:
            media_name = self.yt.title
        if media_name.endswith('.mp4') or media_name.endswith('.wav'):
            media_name = media_name[:-4]
        self.media_name = media_name
        self.diarization = None
        self.audio_filepath = None
        self.video_filepath = None
        self.audio_filepath_wav = None
        self.full_video_filepath = None

        self.total_duration = float(self.yt.vid_info['videoDetails']['lengthSeconds'])
        self.start_at = start_at if start_at else 0
        self.end_at = min(end_at,self.total_duration) if end_at else self.total_duration
        self.video_events = None
        self.caption_store = {}
        # self.pyannote_audio = Audio()
        # self.audio_file = str(TEMP_DIR / f'{media_name}.mp3')

    def __del__(self):
        try:
            if self.audio_filepath:
                os.remove(self.audio_filepath)
        except:
            pass
        try:
            if self.video_filepath:
                os.remove(self.video_filepath)
        except:
            pass
        try:
            if self.audio_filepath_wav:
                os.remove(self.audio_filepath_wav)
        except:
            pass
        try:
            if self.full_video_filepath:
                os.remove(self.full_video_filepath)
        except:
            pass


    def get_all_captions(self):
        return [caption_to_dict(c) for c in list(self.yt.captions.keys())]


    def get_video_streams(self):
        return self.yt.streams
    

    def get_caption_transcript(self, code):
        caption_obj = self.yt.captions[code]
        if '(auto-generated)' in caption_obj.name:
            print('auto generated captions')
            captions_json = xml_auto_caption_to_json(caption_obj.xml_captions)
            processed_captions_json = process_auto_caption_json(captions_json, self.start_at, self.end_at)
        else:
            #! TODO: this code is buggy. the xml_caption_to_json gives diffrent results than the auto caption
            print('no auto captions')
            processed_captions_json = xml_caption_to_json(caption_obj.xml_captions)
        return processed_captions_json


    def get_captions(self, code):
        # try:
        # if self.diarization is None:
            # raise MediaGenerationError('Diarization not found')
            # audio_filepath = self.get_audio()
            # self.diarization = get_speaker_diarization(audio_filepath)
        self.yt.streams
        c = self.yt.captions[code]
        cap = YTCaptions(c, self.diarization, self.audio_filepath_wav, self.start_at, self.end_at)
        self.caption_store[code] = cap
        return cap
        # except KeyError as e:
        #     raise MediaGenerationError(f'No caption found for {code}')

    def get_video_events(self):
        if not self.video_events:
            if not self.video_filepath:
                self.get_video()
            self.video_events = VideoShots(self.video_filepath)
        return self.video_events

    def get_video_streams(self, file_extension='mp4'):
        streams =  self.yt.streams.filter(progressive=True, file_extension=file_extension)
        if not streams:
            print('no progressive streams found')
            streams = self.yt.streams.filter(file_extension=file_extension)

        return streams.order_by('resolution').desc()


    def get_video(self, file_extension='mp4', itag=None):
        video_stream = self.get_video_streams().get_by_itag(itag) if itag else self.get_video_streams().first()
        self.video_filepath = str(TEMP_DIR / f'{self.media_name}.{file_extension}')
        self.full_video_filepath = str(TEMP_DIR / f'{self.media_name}_full.{file_extension}')
        video_stream.download(TEMP_DIR, filename=self.full_video_filepath)
        if os.path.exists(self.video_filepath):
            os.remove(self.video_filepath)
        input_stream = ffmpeg.input(self.full_video_filepath)
        pts = "PTS-STARTPTS"
        video = input_stream.trim(start=self.start_at, end=self.end_at).setpts(pts)
        audio = (input_stream
                .filter_("atrim", start=self.start_at, end=self.end_at)
                .filter_("asetpts", pts))

        video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
        output = ffmpeg.output(video_and_audio, self.video_filepath, format="mp4")
        output.run()
        return self.video_filepath


    def get_audio_streams(self):
        # return self.yt.streams.filter(only_audio=True).order_by('abr').desc().first()
        return self.yt.streams.filter(only_audio=True).order_by('abr').desc()
    


    def get_audio(self, generate_wav=True, itag=None, output_segment=False):
        if self.audio_filepath:
            return self.audio_filepath
        audio_streams = self.get_audio_streams()
        audio_stream = audio_streams.get_by_itag(itag) if itag else audio_streams.first()
        ext = audio_stream.mime_type.split('/')[1]
        self.audio_filepath = str(TEMP_DIR / f'{self.media_name}.{ext}')
        audio_stream.download(TEMP_DIR, filename=self.audio_filepath)
        print('downloaded. ')
        seg = AudioSegment.from_file(TEMP_DIR /  self.audio_filepath, ext)

        if output_segment:
            return seg
        # seg[int(start_at * 1000):int(end_at * 1000)].export(self.audio_filepath, tags={'durations': json.dumps(transcript)})
        seg[int(self.start_at * 1000):int(self.end_at * 1000)].export(self.audio_filepath)
        print('trimmed.')
        if ext != 'wav' and generate_wav:
            print('exporting to wav')
            audio = AudioSegment.from_file(self.audio_filepath)
            self.audio_filepath_wav = str(TEMP_DIR / f'{self.media_name}.wav')
            audio.export(self.audio_filepath_wav, format="wav")
        else:
            self.audio_filepath_wav = self.audio_filepath
        return self.audio_filepath


    def export_audio_with_transcript(self, transcript, file_extension='mp3'):        
        transcript = {}
        for code in self.get_all_captions():
            transcript[code] = self.get_captions(code).to_json()
        seg = AudioSegment.from_file(self.audio_filepath)
        seg.export(self.audio_filepath, tags={'durations': json.dumps(transcript)})
        return self.audio_filepath


    def sub_track(self, start_at=None, end_at=None, use_wav=True):
        if use_wav:
            wavform, rate = librosa.load(self.audio_filepath_wav)
        else:
            wavform, rate = librosa.load(self.audio_filepath)
        ipd.display(ipd.Audio(wavform[int(start_at * rate):int(end_at * rate)], rate=rate))    


    


def caption_to_dict(c: Caption):
    return {
        'name': c.name,
        'code': c.code,
        'url': c.url,
        # 'captions': c
        # 'xml_captions': c.generate_srt_captions(),
        # 'json_captions': xml_caption_to_json(c, c.generate_srt_captions())
    }









def transcript_sentences_to_words(processed_transcript):
    words = []
    for sentence in processed_transcript:
        for w in sentence['words']:
            words.append({
                'word': w['word'],
                'punct_word': '',
                'duration': w['duration'],
                'time': w['time'],
    })
    return words



def get_full_text(processed_transcript):
    words = []
    for sentence in processed_transcript:
        for w in sentence['words']:
            words.append(w['word'])
    full_text = ' '.join(words)
    return full_text


def get_audio_with_word_transcript(yt, transcript, video_filename, start_at, end_at):
    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
    filename = video_filename[:-4] if video_filename.endswith('.mp4') else video_filename
    ext = audio_stream.mime_type.split('/')[1]
    audio_filename = filename + '.' + ext
    audio_stream.download(TEMP_DIR, filename=audio_filename)
    for t in transcript:
        processed_transcript = process_auto_caption_json(t['transcript'], start_at, end_at)
        full_text = get_full_text(processed_transcript)        
        punct_text, j = add_punctuation(full_text, int(len(full_text.split(' ')) * 1.3333) + 20)
        processed_transcript_words = transcript_sentences_to_words(processed_transcript)
        t['transcript'] = merge_punctuation(processed_transcript_words, punct_text)
    seg = AudioSegment.from_file(TEMP_DIR /  audio_filename, ext)
    
    seg[int(start_at * 1000):int(end_at * 1000)].export(TEMP_DIR / audio_filename, tags={'durations': json.dumps(transcript)})
    # seg.export(TEMP_DIR / audio_filename, tags={'durations': json.dumps(transcript)})
    return audio_filename


def download_youtube(video_filename, yt):
    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not video_stream:
        raise MediaGenerationError('no video stream found')
    filename = video_filename[:-4] if video_filename.endswith('.mp4') else video_filename
    video_stream.download(TEMP_DIR, filename=filename+'_full.mp4')
    full_video_path = str(TEMP_DIR / (filename+'_full.mp4'))
    # ffmpeg_extract_subclip(full_video_path, start_at, end_at, targetname=str(TEMP_DIR / video_filename))
    # os.remove(full_video_path)
    return full_video_path, video_stream.fps, video_stream.resolution


def trim_video(video_filename, full_video_path, start_at, end_at):    
    filename = video_filename[:-4] if video_filename.endswith('.mp4') else video_filename
    outpath = str(TEMP_DIR / (filename + '.mp4'))
    if os.path.exists(outpath):
        os.remove(outpath)
    input_stream = ffmpeg.input(full_video_path)
    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start_at, end=end_at).setpts(pts)
    audio = (input_stream
            .filter_("atrim", start=start_at, end=end_at)
            .filter_("asetpts", pts))

    video_and_audio = ffmpeg.concat(video, audio, v=1, a=1)
    output = ffmpeg.output(video_and_audio, outpath, format="mp4")
    output.run()
    return filename + '.mp4'


def get_youtube(url):
    yt = YouTube(url)    
    lang_tags = [t.code for t in list(yt.captions.keys())]
    transcripts = []
    # for lt in lang_tags:
    #     caption = yt.captions[lt]
    #     tran_json, segments = xml_caption_to_json(caption, caption.xml_captions)
    #     transcripts.append({
    #         'lang': lt,
    #         'transcript': tran_json
    #     })
    for c in yt.captions.all():
        if '(auto-generated)' in c.name:
            transcript = xml_auto_caption_to_json(c.xml_captions)
            transcripts.append({
                'name': c.name,
                'code': c.code,
                'lang': c.name,
                'url': c.url,
                'transcript': transcript
            })
    return yt, transcripts