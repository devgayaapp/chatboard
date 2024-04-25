import io
from pathlib import Path
import requests
from components.video.video_shots import Shot, VideoShots
from components.image.image import Image
from config import TEMP_DIR
from util.boto import upload_s3_file
import hashlib
from urllib.parse import urlparse

try:
    import cv2
    import PIL.Image as pimg
    from moviepy.editor import VideoFileClip
except ModuleNotFoundError:
    pimg = None
    print("cv2 and PIL are not installed. Image class will not work.")
    pass






def get_url_video_extension(url):
    # Parse the URL
    parsed_url = urlparse(url)
    # Extract the path component
    path = parsed_url.path
    # Extract the extension
    extension = path.split('.')[-1]
    return extension

def get_url_video_filename(url):
    # Parse the URL
    parsed_url = urlparse(url)
    # Extract the path component
    path = parsed_url.path
    filename = path.split('/')[-1]
    return filename




def fetch_url(url, filename, verify):
    headers = {'User-Agent': 'Musbot/0.1'}
    response = requests.get(url, headers=headers, verify=verify, stream=True)
    if response.status_code != 200:
        raise Exception(f'image fetching is not succesfull: {response.reason}')    
    # ext = get_url_video_extension(url)
    if filename is None:
        filename = get_url_video_filename(url)
    # filename = hashlib.md5(url.encode('utf-8')).hexdigest() + '.' + ext    
    filepath = TEMP_DIR / filename
    with open(filepath, 'wb') as f:
        f.write(response.content)
    return filepath



class Video():


    def __init__(self, filepath, width=None, height=None, duration=None, lazy=False, shots: VideoShots=None) -> None:
        self.lazy = lazy
        self._is_downloaded = False
        if type(filepath) == str:
            filepath = Path(filepath)
        self.filepath = filepath        
        self.shots = shots
        self._video_clip = VideoFileClip(str(self.filepath))
        self.width = self.video_clip.size[0]
        self.height = self.video_clip.size[1]
        self.duration = self.video_clip.duration
        self.fps = self.video_clip.fps
        self.size = self.filepath.stat().st_size 
        # self.resized_filepath = None      
    
    def add_shot(self, shot: Shot):
        if self.shots is None:
            self.shots = []
        self.shots.append(shot)
        

    def _file_to_cv2(self, filepath):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_video_clip' in state and state['_video_clip'] is not None:
            del state['_video_clip']
            state['_video_clip'] = None
        return state
    
    @property
    def video_clip(self):
        self._init_video_clip()
        return self._video_clip
    
    @video_clip.setter
    def video_clip(self, value):
        self._video_clip = value

    @staticmethod
    def from_youtube(url):
        pass

    @staticmethod
    def from_url(url, filename=None):
        filepath = fetch_url(url, filename=filename, verify=False)
        return Video(filepath)


    @staticmethod
    def from_file(filepath):
        return Video(
            filepath=filepath,
        )

    @staticmethod
    def from_bytes(bytes):
        pass
    
    def to_s3(self, bucket, filename):
        upload_s3_file(self.filepath, bucket, filename)

    def delete(self):
        if hasattr(self, 'video_clip') and self.video_clip is not None:
            self.video_clip.close()
            self.video_clip = None
        self.filepath.unlink()
        # if self.resized_filepath:
            # self.resized_filepath.unlink()

    def _init_video_clip(self):
        if self._video_clip is None:
            self._video_clip = VideoFileClip(str(self.filepath))

    def get_frame(self, t):
        # self.init_video_clip()
        np_arr = self.video_clip.get_frame(t)
        return Image.from_numpy(np_arr)


    def get_thumbnail(self):
        if self.shots:
            thumbnail_img = self.shots[0].get_image()
        else:
            thumbnail_img = self.get_frame(0)
        return thumbnail_img.get_thumbnail()


    # .subclip(50,60)
    def subclip(self, start, end):
        # self.init_video_clip()
        return self.video_clip.subclip(start, end)

    def _repr_png_(self):
        if self.shots:
            return self.shots[0].get_image().pil_img._repr_png_()
        else:
            return None
            
    def show(self):
        # self.init_video_clip()
        return self.video_clip.ipython_display(width=480)
    
    def downsize(self):
        # self.init_video_clip()
        self.resized_filepath = self.filepath.parent / ("resized_" + self.filepath.name)
        if self.height > self.width:
            resized_video = self.video_clip.resize(width=360)
        else:
            resized_video = self.video_clip.resize(height=360)
        resized_video.write_videofile(str(self.resized_filepath))
        return self.resized_filepath
