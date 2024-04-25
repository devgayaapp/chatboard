
from os import execv
import tempfile
from typing import List, Tuple
from click import BadParameter
import cv2
from util.types import ImageSize, Coordinates
from components.paragraph import Media
import numpy as np
from config import TEMP_DIR
import math
# from components.video.video import VideoChannelScreen


class BadTextParameters(Exception):
    def __init__(self):
        pass

class VideoChannelScreen: 

    def __init__(self, filename=None):
        self.filename = filename
        self.file = None
        self._open_file(None)
        self.input = None

    def _open_file(self, openfile):
        if not self.filename:
            self.file = tempfile.NamedTemporaryFile(suffix='.avi', dir=TEMP_DIR)
            self.filename = self.file.name 

    def get_input(self):
        raise DeprecationWarning('ffmpeg is not used anymore')
        self.input = ffmpeg.input(self.filename)
        return self.input

    def close(self):
        if self.file.close:
            self.file.close()

    def __del__(self):
        if self.file.close:
            self.file.close()

    

class VideoWriter():

    def __init__(self, fps: int, cap_size: ImageSize):
        self._fps: int = fps
        self._cap_size: ImageSize = cap_size
        self._title: str = None
        self._text: str = None
        self._images: List[Media] = []
        self._background = None
        self._fourcc: int = None
        self._is_initialized: bool = False
        self._cv_video_writer = None
        self._duration: int = None

    @property
    def cv_cap_size(self):
        return (self._cap_size[1], self._cap_size[0])

    def _cleanup(self):
        self._title = None
        self._title_font_scale = 3
        self._title_thickness = 4
        self._text = None
        self._images = []
        self._background = None
        self._fourcc = None
        self._is_initialized = False
        self._cv_video_writer = None
        self._duration = None
        self._current_channel = None
         

    def start(self, filename=None):
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.is_initialized = True
        self._cv_video_writer = cv2.VideoWriter()        
        self._current_channel = VideoChannelScreen(filename)
        self._cv_video_writer.open(
            self._current_channel.filename, 
            self._fourcc, 
            self._fps, 
            self.cv_cap_size
        )
        return self

    def background(self, color='black'):
        return self

    def title(self, title: str, font_scale: float = 1, thickness: float=2):
        self._title = title
        self._title_font_scale = font_scale
        self._title_thickness = thickness
        return self

    def images(self, images: Media):
        self._images = images
        return self

    def duration(self, duration: int):
        self._duration = duration
        return self

    def animation(self, type: str):
        return self

    def sanitize_image(self, img: np.ndarray) -> np.ndarray:
        (screen_height, screen_width) = self._cap_size
        if screen_width < screen_height:
            screen_ratio = screen_height / screen_width
            img_ratio = img.size[0] / img.size[1]
            if screen_ratio > img_ratio:
            # img_ratio = img.shape[1] / img.shape[0]                
                new_img_width = int((img.size[0] * screen_width) / screen_height)
                if new_img_width > img.size[1]:
                    img = img.resize((-1, new_img_width))
                # img._mat = img.mat[:, img_center - math.floor(new_img_width / 2): img_center + math.ceil(new_img_width / 2)]
                # cropped_img = img.resize(self._cap_size)
                img_center = int(img.size[1] / 2)
                cropped_img = img.crop(0, img.size[0], img_center - math.floor(new_img_width / 2), img_center + math.ceil(new_img_width / 2))
                resized_img = cropped_img.resize(self._cap_size)
                return resized_img
            else:                
                new_img_height = int((img.size[1] * screen_height) / screen_width)
                if new_img_height > img.size[0]:
                    img = img.resize((new_img_height, -1))
                    # img = img.resize((new_img_height, int((img.size[1] * new_img_height) / img.size[0])))
                # cropped_mat = img.mat[img_center - math.floor(new_img_height / 2): img_center + math.ceil(new_img_height / 2 ), :]
                # new_img = img.copy()
                img_center = int(img.size[0] / 2)
                cropped_img = img.crop(img_center - math.floor(new_img_height / 2), img_center + math.ceil(new_img_height / 2 ), 0, img.size[1])
                resized_img = cropped_img.resize(self._cap_size)
                return resized_img
        else:
            raise Exception('Image sanitization is not implemented for width screen movies')

    def _write_images(self):
        time_per_image = self._duration / len(self._images)
        for image in self._images:
            if image.cv_size != self.cv_cap_size:
                sanitized_image = self.sanitize_image(image)
            for _ in range(round(time_per_image * self._fps)):
                self._cv_video_writer.write(sanitized_image.mat)

    def _write_background(self):
        img = self._create_color_image(self._cap_size, (0,0,0))
        # position = (100, int(self._cap_size[1] / 2))
        if self._title:
            self._write_text(self._title, img, self._title_font_scale, self._title_thickness)
        for _ in range(round(self._duration * self._fps)):
            self._cv_video_writer.write(img)
        return self

    def _write_text(self, text: str, img: np.ndarray, font_scale: float=3, thickness: float=4):
        SIDE_LINE_WIDTH = 30
        LINE_SPACEING = 30
        #calculate the size of the font
        # https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
        font = cv2.FONT_HERSHEY_DUPLEX
        lines = []
        words = text.split(' ')

        def get_size(text):
            font_size_res = cv2.getTextSize(text, fontFace=font, fontScale=font_scale, thickness=thickness)
            return (font_size_res[0][1], font_size_res[0][0])

        def close_line(lines):
            lines.append({'text': '', 'size': (0,0)})
            line = lines[-1]
            return line
        
        max_line_length = img.shape[1] - 2 * SIDE_LINE_WIDTH

        current_line = close_line(lines)
        while len(words):
            w = words[0]
            line_size = get_size(current_line['text'] + w)
            if line_size[1] < max_line_length:
                current_line['text'] += words.pop(0) + ' '
                current_line['size'] = line_size
            if  current_line['text'] == '' and line_size[1] > max_line_length:
                raise BadTextParameters('Can not fit word into the screen')
            else:
                current_line = close_line(lines)


        total_height = len(lines) * lines[0]['size'][0] + (len(lines)-1) * LINE_SPACEING
        max_row_width = max(lines, key=lambda t :t['size'][1])['size'][1]
        row_height = lines[0]['size'][0] + LINE_SPACEING
        left = int((img.shape[1] - max_row_width)/2)

        for i, row in enumerate(lines):
            #width, height
            top = int((img.shape[0] - total_height)/2)
            top += i * row_height
            position = (left, top)
            cv2.putText(
                img=img, 
                text=row['text'], 
                org=position, 
                fontFace=font, 
                fontScale=font_scale, 
                color=(255, 255, 255),
                thickness=thickness)

    def _create_color_image(self, size: ImageSize, color: Tuple[int, int, int]) -> np.ndarray:
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:,:,0] = color[0]
        img[:,:,1] = color[1]
        img[:,:,2] = color[2]
        return img

    def _release(self):
        self._cv_video_writer.release()

    def write(self) -> VideoChannelScreen:
        if self._images:
            self._write_images()
        else:
            self._write_background()
        self._release()
        channel = self._current_channel
        channel.get_input()
        self._cleanup()
        return channel

    



