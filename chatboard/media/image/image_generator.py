
from typing import Callable, List, Optional, Union
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionUpscalePipeline, StableDiffusionImageVariationPipeline
from diffusers import StableDiffusionPipeline, DiffusionPipeline, EulerDiscreteScheduler
import random
import torch

from components.image.image import Image
from components.image.generated_image import GeneratedImage




DEFAULT_NEGATIVE_PROMPT = ", duplication artifact, deformed eyes, extra limb"


def spread_list(prompt: Union[str, List[str], int, list[int]], n: int) -> List[str]:
    """Spread a prompt over n images"""
    if isinstance(prompt, str) or isinstance(prompt, int):
        prompt = [prompt]
    if len(prompt) == 1:
        return prompt * n
    else:
        return prompt

class ImageGenerator:


    def __init__(self) -> None:
        self.inpaint_pipe = None
        self.text_to_image_pipe = None
        self.image_to_image_pipe = None
        self.upscale_pipe = None
        self.current_pipe = None
        self.variations_pipe = None
        pass


    def unload_current(self):
        if self.current_pipe:
            if self.current_pipe == "inpaint":
                self.inpaint_pipe.to("cpu")
            elif self.current_pipe == "text_to_image":
                self.text_to_image_pipe.to("cpu")
            elif self.current_pipe == "image_to_image":
                self.image_to_image_pipe.to("cpu")
            elif self.current_pipe == "upscale":
                self.upscale_pipe.to("cpu")
            elif self.current_pipe == "variations":
                self.variations_pipe.to("cpu")


    def load_stable_diffusion_inpainting(self):
        if not self.inpaint_pipe:
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
                requires_safety_checker=False, 
                safety_checker=None
            )

        if self.current_pipe != "inpaint":
            self.unload_current()
            self.current_pipe = "inpaint"
            self.inpaint_pipe.to("cuda")


    def load_stable_diffusion(self):
        if not self.text_to_image_pipe:
            model_id = "stabilityai/stable-diffusion-2"
            # model_id = "stabilityai/stable-diffusion-2-1"
            # scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
            # pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None)
            self.text_to_image_pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, requires_safety_checker=False, safety_checker=None)
        
        if self.current_pipe != "text_to_image":
            self.unload_current()
            self.current_pipe = "text_to_image"
            self.text_to_image_pipe.to("cuda")


    def load_image_to_image_pipeline(self):
        if not self.image_to_image_pipe:
            self.image_to_image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                torch_dtype=torch.float16,
                requires_safety_checker=False, 
                safety_checker=None
                )

        if self.current_pipe != "image_to_image":
            self.unload_current()
            self.current_pipe = "image_to_image"
            self.image_to_image_pipe.to("cuda")

    def load_upscale_pipeline(self):
        if not self.upscale_pipe:
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            self.upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)

        if self.current_pipe != "upscale":
            self.unload_current()
            self.current_pipe = "upscale"
            self.upscale_pipe.to("cuda")


    def load_variations_pipeline(self):
        if not self.variations_pipe:
            self.variations_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
                "lambdalabs/sd-image-variations-diffusers",
                revision="v2.0",
                requires_safety_checker=False, 
                safety_checker=None
            )

        if self.current_pipe != "variations":
            self.unload_current()
            self.current_pipe = "variations"
            self.variations_pipe.to("cuda")


    def gen_seed_list(num_seeds=4):
        return [random.randint(0, 2**32-1) for _ in range(num_seeds)]

    def gen_generator_list(seed_list):
        return [torch.Generator("cuda").manual_seed(seed) for seed in seed_list]

    def preprocess_params(prompt, negative_prompt, num_images, seed):
        prompt = spread_list(prompt, num_images)
        if negative_prompt:
            negative_prompt = spread_list(negative_prompt, num_images)

        num_images = len(prompt)
             
        if not seed:
            seed_list = ImageGenerator.gen_seed_list(num_images)            
        if seed:
            seed_list = spread_list(seed, num_images)

        generator = ImageGenerator.gen_generator_list(seed_list)

        if negative_prompt:        
            if type(negative_prompt) == str:
                negative_prompt+= ', ' + DEFAULT_NEGATIVE_PROMPT
                negative_prompt = [negative_prompt] * num_images 
            else:
                negative_prompt = [a+b for a,b in zip(negative_prompt, [DEFAULT_NEGATIVE_PROMPT] * num_images)]
        return prompt, negative_prompt, num_images, seed_list, generator



    def image_to_image(self, prompt, image, negative_prompt=DEFAULT_NEGATIVE_PROMPT, num_images=2, seed_list=None, num_inference_steps=50, strength=0.75, guidance_scale=7.5):
        self.load_image_to_image_pipeline()
        prompt_list = [prompt] * num_images
        negative_prompt_list = None
        if not seed_list:
            seed_list = ImageGenerator.gen_seed_list(num_images)

        generator = ImageGenerator.gen_generator_list(seed_list)

        if negative_prompt:
            negative_prompt += ', ' + DEFAULT_NEGATIVE_PROMPT
            negative_prompt_list = [negative_prompt] * num_images
        out = self.image_to_image_pipe(
            prompt=prompt_list, 
            negative_prompt=negative_prompt_list,
            generator=generator,
            num_inference_steps=num_inference_steps,
            image=image.pil_img, 
            strength=strength, 
            guidance_scale=guidance_scale
            )
        image_list = [Image.from_pil(img) for img in out.images]
        return image_list, seed_list
    

    def inpaint(self, prompt: Union[str, List[str]], mask_image: Image, init_image: Image, negative_prompt: Union[str, List[str]]=None, num_images=4, seed: Union[int, List[int]]=None, num_inference_steps=50, size=768):
        self.load_stable_diffusion_inpainting()
        prompt, negative_prompt, num_images, seed_list, generator = ImageGenerator.preprocess_params(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                seed=seed
            )
        
        out = self.inpaint_pipe(
            prompt=prompt, 
            negative_prompt=prompt,
            image=init_image.pil_img, 
            mask_image=mask_image.pil_img,
            # height=size,
            # width=size,
            num_inference_steps=num_inference_steps,
            generator=generator
        )
        image_list = [Image.from_pil(img) for img in out.images]     
        return image_list, seed_list


    def generate(self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]=None, num_images: int=4, seed: int=None, num_inference_steps: int=50, size:int=768):
        self.load_stable_diffusion()
        #comics
        # prompt += ", high quality portrait, art by makoto shinkai, Ultra hd! Realistic! helmut lang, beautiful face, medium format, fuji superia 400, iso 400, surrealistic, 8k"
        # realistic
        # prompt += ", pexels contest winner, hurufiyya, loosely cropped, acting headshot: 6 | fox in a lab coat, extra limb, from scene from twin peaks, brutalist futuristic interior, retro futurism, dramatic nautical scene, ornate hospital room, crumbling masonry, pale blue armor, mechanical paw, laser guns, pulp sci fi, two deer wearing suits : -2"
        # prompt += ", Ultra hd! Realistic! helmut lang, beautiful face, medium format, fuji superia 400, iso 400, surrealistic, 8k"

        prompt, negative_prompt, num_images, seed_list, generator = ImageGenerator.preprocess_params(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                seed=seed
            )


        out = self.text_to_image_pipe(
            prompt= prompt,
            negative_prompt=negative_prompt,
            height=size,
            width=size,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        image_list = [Image.from_pil(img) for img in out.images]
        return image_list, seed_list



    def upscale(self, prompt, image, negative_prompt=None, num_inference_steps: int = 75, guidance_scale: float = 9.0, noise_level: int = 20):
        self.load_upscale_pipeline()
        out = self.upscale_pipe(
            prompt=prompt, 
            image=image.pil_img,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_level=noise_level
            )
        Image.from_pil(out.images[0])
        return Image


    def variations(self, image, guidance_scale=3, num_images_per_prompt=4, num_inference_steps=50):
        self.load_variations_pipeline()
        out = self.variations_pipe(
            image=image.pil_img,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images_per_prompt,
            guidance_scale=guidance_scale,
        )
        image_list = [Image.from_pil(img) for img in out.images]
        return image_list