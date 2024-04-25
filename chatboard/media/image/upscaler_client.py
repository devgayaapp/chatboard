import asyncio
from bentoml.client import Client
from components.image.image import Image
from components.image.generated_image import GeneratedImage
from config import AI_MEDIA_PROCESSING_URL
import json




class ProcessorServerException(Exception):
    pass


class UpscalerClient:

    def __init__(self, url=AI_MEDIA_PROCESSING_URL) -> None:
        self._server_url = url
        try:
            self._client = Client.from_url(url)
        except Exception as e:
            self._client = None
            print(f"Media processor is not available at {url}")

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = Client.from_url(self._server_url)
            except Exception as e:
                raise ProcessorServerException(f"processor server is not available at {self._server_url}")
        return self._client

    # def improve_image(self, image, fidelity_weight=0.5, upscale=2):
    #     res = self.client.improve_image(
    #         img= image.pil_img,
    #         data ={
    #             'fidelity_weight': fidelity_weight,
    #             'upscale': upscale
    #         }
    #     )
    #     return self.unpack_response(image, res)


    async def improve_image(self, image, fidelity_weight=0.5, upscale=2):
        # res = await self.client.async_improve_image(
        #     img= image.pil_img,
        #     data ={
        #         'fidelity_weight': fidelity_weight,
        #         'upscale': upscale
        #     },
        # )
        res = await asyncio.to_thread(
            self.client.improve_image,
            img= image.pil_img,
            data ={
                'fidelity_weight': fidelity_weight,
                'upscale': upscale
            },
        )
        return self.unpack_response(image, res)

        
    def unpack_response(self, source_image, res):
        img = res["img"]
        data = res["data"]

        if type(source_image) == Image:
            image = Image.from_numpy(img)
        elif type(source_image) == GeneratedImage:
            image = GeneratedImage(
                numpy_arr=img,
                opacity=False,
                prompt=source_image.prompt,
                seed=source_image.seed,
                height=img.shape[0],
                width=img.shape[1],
                steps=source_image.steps,
                cfg=source_image.cfg,
                canny_image = source_image.canny_image,
                pose_image = source_image.pose_image,
                controlnet_conditioning_scale=source_image.controlnet_conditioning_scale,
                filename=source_image.filename,
                control_data = source_image.control_data
            )
        else:
            raise Exception("Unknown image type")
        
        return image, data