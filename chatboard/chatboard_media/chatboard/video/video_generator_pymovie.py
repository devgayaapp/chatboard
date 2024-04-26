
from os import execv
import re
import tempfile
from typing import List, Tuple
from click import BadParameter
import cv2
from components.animations.image_animations import horizontal_scroll, resize, resize_frame, rotating_resizing, translate, translate_scroll
from components.animations.text_animations import moveLetters,cascade, vortex, arrive, vortexout, reveal
from util.types import ImageSize, Coordinates
from components.paragraph import Media
import numpy as np
from config import TEMP_DIR
import math
# from moviepy.editor import ImageClip, TextClip, ColorClip, CompositeVideoClip, concatenate_videoclips
# from moviepy.video.tools.segmenting import findObjects

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

    

class PymovieVideoWriter():

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
        self._image_screen_height = int(0.7 * self._cap_size[0])
        self._is_create_subtitles = False
        self._text = None     
        self.subtitles_sentances = [] 


    @property
    def cv_cap_size(self):
        return (self._cap_size[1], self._cap_size[0])

    def _cleanup(self):
        self.clip = None
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
        self._is_create_subtitles = False  
        self._text = None
        self.subtitles_sentances = [] 
        

    def start(self, filename=None):
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.is_initialized = True
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

    def subtitles(self, text):
        self._is_create_subtitles = True
        self._text = text
        text = self._text
        splits = re.split(', |_|\.|-|!|\+', text)
        self.subtitles_sentances = []
        time_per_word = self._duration / len(text.split(' '))
        for s in splits:
            words = s.split(' ')
            window = 5
            for i in range(math.ceil(len(words) / window)):
                sent = words[i*window:(i+1)*window]
                self.subtitles_sentances.append({
                    'text': ' '.join(sent), 
                    'length': len(sent),
                    'duration': time_per_word * len(sent)
                    })
        return self

    def animation(self, type: str):
        return self

    def centrelize(self, img: np.ndarray, cap_size: ImageSize) -> np.ndarray:
        (screen_height, screen_width) = cap_size
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
                resized_img = cropped_img.resize(cap_size)
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
                resized_img = cropped_img.resize(cap_size)
                return resized_img
        else:
            raise Exception('Image sanitization is not implemented for width screen movies')

    def resize_image(self, img: np.ndarray, cap_size: ImageSize) -> np.ndarray:
        (screen_height, screen_width) = cap_size
        (img_height, image_width) = img.size
        img_ratio = img_height / image_width
        screen_ratio = screen_height / screen_width
        if img_ratio <= 0.8:
            # new_img_height = int((image_width * screen_height) / screen_width)
            return img.resize((screen_height, -1)), 'horizontal'
            #width picture
        elif img_ratio > 0.8 and img_ratio < 1.2:
            return img, 'fit'
        elif img_ratio >= 1.2:
            resized_img = img.resize((-1, screen_width))            
            return resized_img.crop(0, screen_height, 0, screen_width), 'vertical'
            
            #tall image

    def _write_images(self):
        raise Exception('movie py is no longer in use')

        # time_per_image = self._duration / len(self._images)
        # clips = []
        # for image in self._images:
        #     # if image.cv_size != self.cv_cap_size:
        #         # sanitized_image = self.sanitize_image(image, self._cap_size)
        #     # for _ in range(round(time_per_image * self._fps)):
        #         # self._cv_video_writer.write(sanitized_image.mat)
            
        #     ressized_image, direction = self.resize_image(image, (self._image_screen_height, self._cap_size[1]))
        #     img_clip = ImageClip(ressized_image.img).set_position("top")
        #     screen_size = self._cap_size
        #     slid_in_clip = img_clip.set_duration(1).fl( lambda get_frame, t: rotating_resizing(get_frame, t, 1, 5) )
        #     # slid_out_clip = img_clip.set_duration(3).fl( lambda get_frame, t: translate_scroll(get_frame, t, 3, screen_size[1], screen_size[1]))
        #     d = self._duration            
        #     if direction == 'horizontal':
        #         length = (ressized_image.size[1] - screen_size[1])
        #         # img_stay_clip = img_clip.set_duration(self._duration).fl( lambda get_frame, t : horizontal_scroll(get_frame, t, d, screen_size[1], length))
        #         img_stay_clip = img_clip.set_duration(self._duration).fl( lambda get_frame, t : translate_scroll(get_frame, t, d, (-length, 0)))
        #         slid_out_clip = img_clip.set_duration(1) \
        #             .fl(lambda get_frame, t: translate(get_frame(t), new_position=(-length, 0))) \
        #             .fl(lambda get_frame, t: translate_scroll(get_frame, t, 1, (0, -ressized_image.size[0])))
        #     else:
        #         focus_point = (0 , int(self._cap_size[1] / 2))
        #         target_scale = 1.5
        #         img_stay_clip = img_clip.set_duration(self._duration).fl( lambda get_frame, t: resize(get_frame, t, d, focus_point, target_scale) )
        #         slid_out_clip = img_clip.set_duration(1) \
        #             .fl(lambda get_frame, t: resize_frame(get_frame(t), target_scale, focus=focus_point )) \
        #             .fl(lambda get_frame, t: translate_scroll(get_frame, t, 1, (0, -ressized_image.size[0])))

        #     # clip = concatenate_videoclips([slid_in_clip, img_stay_clip])
        #     # color_clip = ColorClip(self.cv_cap_size, color=(0,0,0)).set_duration(self._duration + 2)
        #     # clip = CompositeVideoClip([color_clip, full_clip.set_position('center')])
        #     clips += [slid_in_clip, img_stay_clip, slid_out_clip]

        # concatinated_clip = concatenate_videoclips(clips)
        # color_clip = ColorClip(self.cv_cap_size, color=(0,0,0)).set_duration(self._duration + 2)
        # # final_clip = CompositeVideoClip([color_clip, concatinated_clip.set_position('center')])
        # if self._is_create_subtitles:
        #     subtitle_clip = self._create_subtitles()
        #     final_clip = CompositeVideoClip([color_clip, concatinated_clip.set_position('left'), subtitle_clip.set_position('top')])

        # self.clip = final_clip
        

    def _write_background(self):
        raise Exception('movie py is no longer in use')

        # img = self._create_color_image(self._cap_size, (0,0,0))
        # # position = (100, int(self._cap_size[1] / 2))
        # if self._title:
        #     title = self._title
        #     txtClip = TextClip(title,color='white', font="Amiri-Bold",
        #         kerning = 5, fontsize=80, align='center', method='caption')
        #     screensize = self.cv_cap_size
        #     cvc = CompositeVideoClip( [txtClip.set_pos('center')],
        #                             size=screensize)
        #     letters = findObjects(cvc, rem_thr=100)            
        #     slide_in = CompositeVideoClip( moveLetters(letters, vortex), size=screensize).subclip(0,2)
        #     slide_out = CompositeVideoClip( moveLetters(letters, vortexout), size=screensize).subclip(0,2)
        #     color_clip = ColorClip(screensize, color=(0,0,0))
        #     stay_clip = CompositeVideoClip([txtClip.set_pos('center')], size=screensize).set_duration(self._duration)
        #     self.clip = concatenate_videoclips([slide_in, stay_clip, slide_out])
        #     # self.clip = concatenate_videoclips([slide_out])
        # return self

    def _create_subtitles(self):
        raise Exception('movie py is no longer in use')

        # text = self._text
        # subtitle_screen_size = int((self._cap_size[0] - self._image_screen_height) / 2)
        # # splits = re.split(', |_|\.|-|!|\+', text)
        # # sentances = []
        # # for s in splits:
        # #     words = s.split(' ')
        # #     window = 5
        # #     for i in range(math.ceil(len(words) / window)):
        # #         sent = words[i*window:(i+1)*window]
        # #         sentances.append({'text': ' '.join(sent), 'length': len(sent)})

        # subtitle_clips = []
        # screensize = self.cv_cap_size
        # time_per_word = self._duration / len(text.split(' '))
        # for s in self.subtitles_sentances:
        #     if s['text'] == '':
        #         continue
        #     # clip = TextClip(
        #     # s['text'], color='white', fontsize=80, font='Amiri-Bold', kerning = 5,
        #     # stroke_width=3, method='caption', align='north', size=(screen_size[0], -1))
        #     txtClip = TextClip(s['text'],color='white', font="Amiri-Bold",
        #                 kerning = 5, fontsize=50, align='center').margin(10)
        #     cvc = CompositeVideoClip( [txtClip.set_pos('center')],
        #                                     size=(screensize[0], subtitle_screen_size))
        #     letters = findObjects(cvc, rem_thr=10)
        #     duration = time_per_word * s['length']
        #     clip_anim = CompositeVideoClip( moveLetters(letters, lambda screenpos, i, nletters: reveal(screenpos, i, nletters, duration)),
        #                             size = (screensize[0], subtitle_screen_size)).subclip(0, duration)
        #     subtitle_clips.append(clip_anim)
        # clip = concatenate_videoclips(subtitle_clips)

        # return clip

    def _write_text(self, text: str, img: np.ndarray, font_scale: float=3, thickness: float=4):
        raise Exception('movie py is no longer in use')

        # txt_clip = TextClip(text, fontsize=60, color='white')
        # self.clip = txt_clip
        
        # SIDE_LINE_WIDTH = 30
        # LINE_SPACEING = 30
        # #calculate the size of the font
        # # https://codeyarns.com/tech/2015-03-11-fonts-in-opencv.html
        # font = cv2.FONT_HERSHEY_DUPLEX
        # lines = []
        # words = text.split(' ')

        # def get_size(text):
        #     font_size_res = cv2.getTextSize(text, fontFace=font, fontScale=font_scale, thickness=thickness)
        #     return (font_size_res[0][1], font_size_res[0][0])

        # def close_line(lines):
        #     lines.append({'text': '', 'size': (0,0)})
        #     line = lines[-1]
        #     return line
        
        # max_line_length = img.shape[1] - 2 * SIDE_LINE_WIDTH

        # current_line = close_line(lines)
        # while len(words):
        #     w = words[0]
        #     line_size = get_size(current_line['text'] + w)
        #     if line_size[1] < max_line_length:
        #         current_line['text'] += words.pop(0) + ' '
        #         current_line['size'] = line_size
        #     if  current_line['text'] == '' and line_size[1] > max_line_length:
        #         raise BadTextParameters('Can not fit word into the screen')
        #     else:
        #         current_line = close_line(lines)


        # total_height = len(lines) * lines[0]['size'][0] + (len(lines)-1) * LINE_SPACEING
        # max_row_width = max(lines, key=lambda t :t['size'][1])['size'][1]
        # row_height = lines[0]['size'][0] + LINE_SPACEING
        # left = int((img.shape[1] - max_row_width)/2)

        # for i, row in enumerate(lines):
        #     #width, height
        #     top = int((img.shape[0] - total_height)/2)
        #     top += i * row_height
        #     position = (left, top)
        #     cv2.putText(
        #         img=img, 
        #         text=row['text'], 
        #         org=position, 
        #         fontFace=font, 
        #         fontScale=font_scale, 
        #         color=(255, 255, 255),
        #         thickness=thickness)

    def _create_color_image(self, size: ImageSize, color: Tuple[int, int, int]) -> np.ndarray:
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:,:,0] = color[0]
        img[:,:,1] = color[1]
        img[:,:,2] = color[2]
        return img


    def write(self) -> VideoChannelScreen:
        if self._images:
            self._write_images()
        else:
            self._write_background()

        # channel = self._current_channel
        # channel.get_input()
        clip = self.clip
        self._cleanup()
        return clip

    def download_media(self, path: str):
        for im in self._images:
            im.populate_media(path)
        return self


    def to_json(self):
        ret_json = {}
        ret_json['title'] = self._title
        ret_json['text'] = self._text
        ret_json['images'] = [i.to_json() for i in self._images]
        ret_json['background'] = self._background
        ret_json['duration'] = self._duration
        ret_json['has_subtitles'] = self._is_create_subtitles
        ret_json['subtitles_sentances'] = self.subtitles_sentances
        return ret_json

    



