from bentoml.client import Client
from components.image.image import Image
from components.image.generated_image import GeneratedImage
from config import AI_IMAGE_URL
import numpy as np
import asyncio



def get_black_image(width, height):
    return Image.from_numpy(np.zeros((width,height,3), dtype=np.uint8))


def get_control_scale(control_img=None, background_img=None):
    controlnet_conditioning_scale = [0.0, 0.0]
    if control_img:
        controlnet_conditioning_scale[0] = 1.0
    if background_img:
        controlnet_conditioning_scale[1] = 0.8
    return controlnet_conditioning_scale


class ImageServerException(Exception):
    pass


class ImageGeneratorClient:

    def __init__(self, server_url=AI_IMAGE_URL) -> None:

        self._server_url = server_url
        try:
            self._client = Client.from_url(server_url)
        except Exception as e:
            self._client = None
            print(f"Image generator is not available at {server_url}")

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = Client.from_url(self._server_url)
            except Exception as e:
                raise ImageServerException(f"Image generator server is not available at {self._server_url}")
        return self._client

    async def txt2img_controlnet(self, prompt, control_image="none", controlnet_conditioning_scale=None):
        params = {}
        if controlnet_conditioning_scale:
            params["controlnet_conditioning_scale"] = controlnet_conditioning_scale
        image_out = await asyncio.to_thread(
            self.client.txt2img_controlnet,
            prompt= prompt,
            control_image=control_image,
            params = params
        )
        return Image.from_pil(image_out)
    

    async def txt2img(self, prompt, params=None):
        params = params or {}
        image_out = await asyncio.to_thread(
            self.client.txt2img,
            prompt= prompt,
            params=params,
        )
        return Image.from_pil(image_out)


    def generate_image2(self, prompts, negative_prompt=None, pose_img=None, canny_img=None, controlnet_conditioning_scale=None, seed=None, num_images=3, height=768, width=768, steps=50, cfg=7.5, high_noise_frac=0.8):
        gen_res = self.client.txt2img2(
            prompt= prompts,            
            negative_prompt= negative_prompt,
            pose_img= pose_img.pil_img if pose_img is not None else get_black_image(round(width / 2), height).pil_img,
            canny_img= canny_img.pil_img if canny_img is not None else get_black_image(width, height).pil_img,
            params= {
                "seed": seed,
                "steps": steps,
                "num_images": num_images,
                "width": width,
                "height": height,
                "cfg": cfg,
                "high_noise_frac": high_noise_frac,
                "controlnet_conditioning_scale": controlnet_conditioning_scale if controlnet_conditioning_scale else get_control_scale(pose_img, canny_img),
            }
        )
        return self.unpack_response(gen_res)


    async def generate_image_async(self, prompts, negative_prompt=None, pose_img=None, canny_img=None, controlnet_conditioning_scale=None, seed=None, num_images=3, height=768, width=768, steps=50, cfg=7.5, high_noise_frac=0.8):
        gen_res = await self.client.async_txt2img2(
            prompt= prompts,            
            negative_prompt= negative_prompt,
            pose_img= pose_img.pil_img if pose_img is not None else get_black_image(round(width / 2), height).pil_img,
            canny_img= canny_img.pil_img if canny_img is not None else get_black_image(width, height).pil_img,
            params= {
                "seed": seed,
                "steps": steps,
                "num_images": num_images,
                "width": width,
                "height": height,
                "cfg": cfg,
                "high_noise_frac": high_noise_frac,
                "controlnet_conditioning_scale": controlnet_conditioning_scale if controlnet_conditioning_scale else get_control_scale(pose_img, canny_img),
            }
        )
        return self.unpack_response(gen_res, canny_img, pose_img)
        
        

        
    def unpack_response(self, gen_res, canny_img=None, pose_img=None):
        imgs_res = gen_res["imgs"]
        params = gen_res["params"]
        # img_arr = [Image.from_numpy(imgs_res["imgs"][i,:,:,:]) for i in range(imgs_res["imgs"].shape[0])]
        imgs_arr = [
            GeneratedImage(
                numpy_arr=imgs_res[i,:,:,:],
                opacity=False,
                prompt=params[i]["prompt"],
                seed=params[i]["seed"],
                height=params[i]["height"],
                width=params[i]["width"],
                steps=params[i]["steps"],
                cfg=params[i]["cfg"],
                canny_image=canny_img,
                pose_image=pose_img,
                controlnet_conditioning_scale=params[i]["controlnet_conditioning_scale"],
            ) for i in range(imgs_res.shape[0])

        ]
        return imgs_arr