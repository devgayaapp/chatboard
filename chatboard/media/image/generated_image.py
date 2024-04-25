from components.image.image import Image
import numpy as np
import hashlib
import time


class GeneratedImage(Image):


    def __init__(self, numpy_arr, opacity, prompt, seed=None, height=768, width=768, steps=50, cfg=7.5, faces=None, canny_image=None, pose_image=None, control_data=None, controlnet_conditioning_scale=None, filename=None):
        super().__init__(opacity=opacity)
        self.np_arr = numpy_arr if numpy_arr.dtype == np.uint8 else numpy_arr.astype(np.uint8)
        self.prompt = prompt
        self.seed = seed
        self.height = height
        self.width = width
        self.steps = steps
        self.cfg = cfg
        self.img_arr = None
        self.canny_image = canny_image
        self.pose_image = pose_image
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.filename = filename if filename is not None else self.generate_filename()
        self.faces = faces
        self.control_data = control_data
        self._file_name = None


    @property
    def file_name(self):
        if self._file_name is None:
            self._file_name = self.generate_filename()
        return self._file_name

    @staticmethod
    def from_generated_image(generated_image):
        return GeneratedImage(
            numpy_arr=generated_image.np_arr,
            opacity=generated_image.opacity,
            prompt=generated_image.prompt,
            seed=generated_image.seed,
            height=generated_image.height,
            width=generated_image.width,
            steps=generated_image.steps,
            cfg=generated_image.cfg,
            faces=generated_image.faces,
            canny_image=generated_image.canny_image,
            pose_image=generated_image.pose_image,
            controlnet_conditioning_scale=generated_image.controlnet_conditioning_scale,
            control_data = generated_image.control_data,
            filename=generated_image.filename,
        )    


    def gen_data_json(self):
        return {
            'prompt': self.prompt,
            'seed': self.seed,
            'height': self.height,
            'width': self.width,
            'steps': self.steps,
            'cfg': self.cfg,
            'controlnet_conditioning_scale': self.controlnet_conditioning_scale,
        }
    
    def generate_filename(self, extension='png'):
        prefix = hashlib.sha256(self.prompt.encode('utf-8')).hexdigest()
        prefix += f'_s{self.seed}'
        return prefix + time.strftime('%H_%M_%S') + f'.{extension}'


        
        

#? https://stable-diffusion-art.com/prompt-guide/


PHOTOREALISM_STYLE = "best quality, 4k, 8k, ultra highres, raw photo in hdr, sharp focus, intricate texture, skin imperfections, photograph of"
PHOTOREALISM_NEGATIVE_PROMPT = "EasyNegative, worst quality, low quality, normal quality, child, painting, drawing, sketch, cartoon, anime, render, 3d, blurry, deformed, disfigured, morbid, mutated, bad anatomy, bad art"


EMPIRE_STYLE = "backlit, digital painting, concept art, smooth, sharp focus, rule of thirds, dark fantasy,intricate details, art by aleksi briclot and Greg Rutkowski, medium shot, Style by Style-Empire, moonlit street"                                
EMPIRE_STYLE_NEGATIVE_PROMPT = "Deformed, blurry, bad anatomy, disfigured, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, ((mutated hands and fingers)), cartoon, 3d, weird colors, lightning"

class StyleTypes:
    PHOTO_REALISM = 'PHOTO_REALISM'
    EMPIRE = 'EMPIRE'


DIM_LIGHTING = '(dimly lit:1.4) ( hard light:1.2), (volumetric:1.2), well-lit'
BRIGHT_LIGHTING = '(brightly lit:1.4), (soft light:1.2), (volumetric:1.2), well-lit'

class LightingTypes:
    DIM = 'DIM'
    BRIGHT = 'BRIGHT'


def append_style(prompt, style):
    if style == StyleTypes.PHOTO_REALISM:
        return f'{prompt}, {PHOTOREALISM_STYLE}'
    elif style == StyleTypes.EMPIRE:
        return f'{prompt}, {EMPIRE_STYLE}'
    return prompt

def append_lighting(prompt, lighting):
    if lighting == LightingTypes.DIM:
        return f'{prompt}, {DIM_LIGHTING}'
    elif lighting == LightingTypes.BRIGHT:
        return f'{prompt}, {BRIGHT_LIGHTING}'
    return prompt

class ImagePrompt:

    def __init__(
            self, 
            prompt, 
            negative_prompt=None,
            sentence=None, 
            topic=None,
            style_prompt=None,
            seed=None,
            height=768,
            width=768,
            steps=25,
            cfg=7.5,
            style=StyleTypes.PHOTO_REALISM,
            lighting=LightingTypes.BRIGHT,
            controlnet_conditioning_scale=None
        ) -> None:
        self.sentence = sentence
        self._negative_prompt = negative_prompt
        self._prompt = prompt
        self.topic = topic
        self.style_prompt = style_prompt
        self.seed = seed
        self.height = height
        self.width = width
        self.steps = steps
        self.cfg = cfg
        self.style = style
        self.lighting = lighting
        self.controlnet_conditioning_scale = controlnet_conditioning_scale if controlnet_conditioning_scale is not None else [0.0, 0,0]



    @property
    def prompt(self):
        p = append_style(self._prompt, self.style)
        p = append_lighting(p, self.lighting)
        return p
    
    @property
    def negative_prompt(self):
        if self._negative_prompt is not None:
            return self._negative_prompt
        if self.style == StyleTypes.PHOTO_REALISM:
            return PHOTOREALISM_NEGATIVE_PROMPT
        elif self.style == StyleTypes.EMPIRE:
            return EMPIRE_STYLE_NEGATIVE_PROMPT
    
    @property
    def time(self):
        if self.sentence is None:
            return None
        return self.sentence.time
    
    @property
    def duration(self):
        if self.sentence is None:
            return None
        return self.sentence.duration
    

    def to_json(self):
        return {
            'prompt': self.prompt,
            'negative_prompt': self.negative_prompt,
            'sentence': self.sentence.text if self.sentence is not None else None,
            'topic': self.topic,            
            'seed': self.seed,
            'height': self.height,
            'width': self.width,
            'steps': self.steps,
            'cfg': self.cfg,
            'style': self.style,
            'lighting': self.lighting,
            'controlnet_conditioning_scale': self.controlnet_conditioning_scale,
        }
    

    @staticmethod
    def from_config(config):
        return ImagePrompt(
            prompt=config.prompt,
        )
    

    @staticmethod
    def from_text(text):
        return ImagePrompt(
            prompt=text,
        )
    


