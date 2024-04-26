from io import BytesIO
import numpy as np
# from pydub import AudioSegment
from collections import namedtuple

from config import TEMP_DIR
import tempfile

import requests
import hashlib
import os
from config import DEBUG
from util.boto import upload_s3_obj

try:
    import soundfile as sf
    import librosa
except:
    pass


# if DEBUG:
    

try:
    import IPython.display as ipd
    import matplotlib.pyplot as plt
    import ffmpeg
except:
    pass


def get_extension(file_name):
    return file_name.split('.')[-1]


def ext2format(ext):
    if ext == 'mp3':
        return 'mp3'
    elif ext == 'wav':
        return 'wav'
    elif ext == 'webm':
        return 'opus'
    else:
        raise Exception(f'extension {ext} is not supported')


def fetch_audio_url(url, verify):
    headers = {'User-Agent': 'Musbot/0.1'}
    response = requests.get(url, headers=headers, verify=verify)
    if response.status_code != 200:
        raise Exception(f'audio {url} fetching is not succesfull: {response.reason}')        
    return BytesIO(response.content)
    # audio_seg=AudioSegment.from_mp3(BytesIO(response.content))
    # return audio_seg


def video2audio(video_url):
    vio = fetch_audio_url(video_url, False)
    vio.seek(0)
    with tempfile.NamedTemporaryFile(suffix='.mp4', dir=TEMP_DIR) as vf:
        vf.write(vio.read())
        input_stream = ffmpeg.input(vf.name)
        audio_filename = str(hashlib.md5(vf.name.encode()).hexdigest())+'.mp3'
        writer = ffmpeg.output(input_stream.audio, str(TEMP_DIR / audio_filename), format="mp3")
        writer.run()
        audio = Audio.from_file(str(TEMP_DIR / audio_filename))
        os.remove(TEMP_DIR / audio_filename)
        return audio
    

TranscriptError = namedtuple('TranscriptError', ['errors', 'fatal_errors'])


class Audio:

    def __init__(self, numpy_arr: np.ndarray, sample_rate: int, normalized: bool=True, ext: str = 'mp3', transcript = None, transcript_error: TranscriptError = None, url: str = None, file_name: str = None):
        
        # if content_bytes is not None:
        #     if ext == 'mp3':
        #         self.audio_segment = AudioSegment.from_mp3(content_bytes, sample_rate )
        #     elif ext == 'wav':
        #         self.audio_segment = AudioSegment.from_wav(content_bytes)
        #     elif ext == 'webm':
        #         #https://github.com/jiaaro/pydub/issues/257
        #         self.audio_segment = AudioSegment.from_file(content_bytes, codec='opus')
        # elif numpy_arr is not None:
        #     if sample_rate is None:
        #         raise Exception('sample_rate must be provided')
        #     # content_bytes, audio_segment, y = write_to_bytesio(numpy_arr, sample_rate, normalized=normalized, media_format=ext2format(ext))
        #     # self.audio_segment = audio_segment
        #     # content_bytes.seek(0)
        #     channels = numpy_arr.ndim
        #     self.audio_segment = AudioSegment(numpy_arr.tobytes(), sample_width=numpy_arr.dtype.itemsize, frame_rate=sample_rate, channels=channels)
        #     content_bytes = BytesIO(content_bytes)
        # elif audio_segment is not None:
        #     self.audio_segment = audio_segment
        #     content_bytes = BytesIO()
        #     audio_segment.export(content_bytes, format=ext)
        #     content_bytes.seek(0)

        # # info = mediainfo(content_bytes.read())
        # info = None
        self._np_arr = numpy_arr

        single_channel = numpy_arr if numpy_arr.ndim == 1 else numpy_arr[0, :]
        self.duration = single_channel.shape[0] / sample_rate
        self.sample_rate = sample_rate
        self.channels = numpy_arr.ndim
        self.ext = ext
        self.transcript = transcript
        self.url = url
        self.file_name = file_name

    @staticmethod
    def from_numpy(numpy_arr, sample_rate, normalized=True):
        return Audio(numpy_arr, sample_rate, normalized)

    @staticmethod
    def from_file(file_name):
        waveform, sr = librosa.load(str(file_name))
        return Audio(waveform, sr, ext=get_extension(str(file_name)), file_name=file_name)

    @staticmethod
    def from_url(url, verify=True):
        ext = get_extension(url)
        if ext == 'mp4' or ext == 'webm':
            return video2audio(url)
        stream = fetch_audio_url(url, verify=verify)
        waveform, sr = librosa.load(stream)
        return Audio(waveform, sr, ext=get_extension(url))
    
    @staticmethod
    def from_bytes(content_bytes):
        waveform, sr = librosa.load(content_bytes)
        return Audio(waveform, sr)
    

    def trim(self, start, end):
        start = int(start * self.sample_rate)
        end = int(end * self.sample_rate)
        return Audio(self.numpy()[start:end], self.sample_rate, ext=self.ext, url=self.url, file_name=self.file_name)
        # return self.audio_segment[start:end]
    # content = self.audio_segment
        # seg = self.audio_segment
        
        # return Audio(audio_segment=seg[int(start * 1000):int(end*1000)], ext=self.ext)

        # np_arr = self.numpy()
        # start = int(start * self.sample_rate)
        # end = int(end * self.sample_rate)
        # return Audio(numpy_arr=np_arr[start:end], sample_rate=self.sample_rate, ext='wav')

        
    

    def numpy(self):        
        return self._np_arr
    
    def play(self):
        if self.transcript:
            print("\n".join([s['sentence'] for s in self.transcript['transcript']]))
        return ipd.Audio(data=self.numpy(), rate=self.sample_rate)        

    def _repr_html_(self):
        return self.play()
        # return ipd.Audio(self.audio_segment)

    def to_file(self, file_name):
        sf.write(file_name, self.numpy(), self.sample_rate)
        # self.audio_segment.export(file_name, format=format)
    
    def to_io(self, ext):
        buffer = BytesIO()
        sf.write(buffer, self.numpy(), self.sample_rate, format=ext)
        buffer.seek(0)
        return buffer

    def to_s3(self, bucket, file_name):
        buffer = BytesIO()
        ext = get_extension(file_name)
        sf.write(buffer, self.numpy(), self.sample_rate, format=ext)
        buffer.seek(0)
        upload_s3_obj(bucket, file_name, buffer)

    def to_bytes(self):
        buffer = BytesIO()
        sf.write(buffer, self.numpy(), self.sample_rate, format=self.ext)
        buffer.seek(0)
        return buffer.getvalue()

    def transcript_stats(self):
        if self.transcript:
            last_sent = self.transcript['transcript'][-1]
            last_word = last_sent['words'][-1]
            last_char = last_word['chars'][-1]
            print('duration:', self.duration)
            print('last word time:', last_word['time'])
            print('last char time:', last_char['time'])


    def waveform(self):
        fig, ax = plt.subplots(figsize=(32, 10))
        waveform = self.numpy()
        ax.plot(waveform)
        word_start = 0
        sample_rate = len(waveform) / self.duration
        if self.transcript:

            tokens = []
            timesteps = []
            seconds = []
            last_word = None
            for sentence in self.transcript['transcript']:
                for word in sentence['words']:
                    last_word = word
                    if len(tokens) and tokens[-1] != "|":
                        tokens.append("|")
                        timesteps.append(int(word['time'] * sample_rate))
                    for c in word['chars']:
                        tokens.append(c['char'])
                        timesteps.append(int(c['time'] * sample_rate ))
                        seconds.append(c['time'])
                tokens.append("|")
                timesteps.append((last_word['time'] + last_word['duration']) * sample_rate)
            ratio = len(waveform) / len(timesteps)
            for i in range(len(tokens)):
                if i != 0 and tokens[i - 1] == "|":
                    word_start = timesteps[i]
                if tokens[i] != "|":
                    plt.annotate(tokens[i].upper(), (timesteps[i], waveform.max() * 1.02), size=14)
                elif i != 0:
                    word_end = timesteps[i]
                    ax.axvspan(word_start, word_end, alpha=0.1, color="red") #vertical span rectangle

        xticks = ax.get_xticks()
        plt.xticks(xticks, xticks / sample_rate)
        ax.set_xlabel("time (sec)")
        ax.set_xlim(0, waveform.shape[0])
        return waveform


def plot_waveform(waveform: np.ndarray, sample_rate: int):
    # waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = np.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)



def write_to_bytesio(wav, sr, normalized=False, tags=None, media_format='mp3'):    
    x = np.stack(
                [wav, wav],                    
            ).transpose()
    stream = BytesIO()
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
        # y = np.float16(x * 2 ** 15)
    else:
        # y = np.int16(x)
        y = np.float16(x)
    audio_segment = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    if media_format == 'webm':
        # song.export(stream, format="webm", bitrate="320k", tags=tags)
        audio_segment.export(stream, format="webm", tags=tags)
    else:
        audio_segment.export(stream, format="mp3", bitrate="320k", tags=tags)
    return stream, audio_segment, y
