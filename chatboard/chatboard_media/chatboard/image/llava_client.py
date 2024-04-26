import asyncio
from typing import List
import requests
from retry import retry
from components.image.image import Image
from bentoml.client import Client
from config import AI_IMAGE_TO_TEXT_URL

def llama_cpp_complete(prompt, image: Image, image_id=12, n_predict=128):
    res = requests.post('http://localhost:8080/completion', json={
        # "prompt": "USER:[img-12]Describe the image in detail.\nASSISTANT:",
        "prompt": prompt,
        "n_predict": 128,
        "image_data": [
            {
            "data": image.to_base64(),
            "id": image_id
            },
            ]
        })
    res_json = res.json()
    return res_json




class ImageToTextServerException(Exception):
    pass


class LlavaClient(object):
    """LlavaClient is a client for the Llava service."""

    def __init__(self, llava_url=AI_IMAGE_TO_TEXT_URL ,type='bento'):
        """Initialize the LlavaClient.

        Args:
            llava_url: The URL of the Llava service.
            llava_port: The port of the Llava service.
        """
        self._llava_url = llava_url
        try:
            self._client = Client.from_url(llava_url)    
        except Exception as e:
            self._client = None
            print(f"Llava server is not available at {llava_url}")
            # raise ImageToTextServerException(f"Llava server is not available at {llava_url}")

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = Client.from_url(self._llava_url)
            except Exception as e:
                raise ImageToTextServerException(f"Llava server is not available at {self._llava_url}")
        return self._client

    @retry(tries=4, delay=5)
    async def complete(self, prompt: str, image: Image):
        if type(image) == Image or str(type(image)) == str(Image):
            image = image.pil_img
        # res = await self.client.async_image2text(
        #     image=image,
        #     prompt=prompt
        # )
        res = await asyncio.to_thread(self.client.image2text, image=image, prompt=prompt)
        return res
    
    @retry(tries=4, delay=5)
    async def complete_multi(self, prompt: str, images: List[Image]):
        if len(images) != 3:
            raise ImageToTextServerException('images must be a list of 3 images')
        images_to_send = []
        for image in images:
            if type(image) == Image:
                image = image.pil_img
            images_to_send.append(image)
        
        # res = await self.client.async_image2text_multi(
        #     image1=images_to_send[0],
        #     image2=images_to_send[1],
        #     image3=images_to_send[2],
        #     prompt=prompt
        # )
        res = await asyncio.to_thread(
            self.client.image2text_multi, 
            image1=images_to_send[0], 
            image2=images_to_send[1], 
            image3=images_to_send[2], 
            prompt=prompt
        )
        return res
    
    @retry(tries=4, delay=2)
    def get_image_embeddings(self, image: Image):
        res = self.client.image_embeddings({
            'image': image.pil_img
        })
        return res 
    
    @retry(tries=4, delay=2)
    async def aget_image_embeddings(self, image: Image):        
        # res = await self.client.async_image_embeddings(image = image.pil_img)
        try:
            res = await asyncio.to_thread(self.client.image_embeddings, image=image.pil_img)
            return res
        except Exception as e:
            print('error in aget_image_embeddings:', e)
            raise e
    
    @retry(tries=4, delay=2)
    def get_text_embeddings(self, text: str):
        res = self.client.text_embeddings({
            'text': text
        })
        return res
    
    @retry(tries=4, delay=2)
    async def aget_text_embeddings(self, text: str):
        try:
            res = await asyncio.to_thread(self.client.text_embeddings, {'text': text})
            return res
        except Exception as e:
            print('error in aget_text_embeddings:', e)
            raise e