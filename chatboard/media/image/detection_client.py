from typing import Union
import requests
from components.image.image import Image
from bentoml.client import Client
import numpy as np
import PIL.Image as PImage
import asyncio

from config import AI_DETECTION_URL


class DetectionServerException(Exception):
    pass


class DetectionClient(object):
    """LlavaClient is a client for the Llava service."""

    def __init__(self, server_url=AI_DETECTION_URL, type='bento'):
        """Initialize the LlavaClient.

        Args:
            llava_url: The URL of the Llava service.
            llava_port: The port of the Llava service.
        """
        self._server_url = server_url
        try:
            self._client = Client.from_url(server_url)
        except Exception as e:
            self._client = None
            print(f"Detection server is not available at {server_url}")
            # raise DetectionServerException(f"Detection server is not available at {server_url}")

    @property
    def client(self):
        if self._client is None:
            try:
                self._client = Client.from_url(self._server_url)
            except Exception as e:
                raise DetectionServerException(f"Detection server is not available at {self._server_url}")
        return self._client


    def recognize_faces(self, image: Union[Image, PImage.Image, np.ndarray]):
        if type(image) == Image:
            image = image.pil_img
        if isinstance(image, PImage.Image):
            image = np.array(image)
        res = self.client.recognize_faces(image)        
        return self.unpack_response(res)
    

    async def arecognize_faces(self, image: Union[Image, PImage.Image, np.ndarray]):
        if type(image) == Image:
            image = image.pil_img
        if isinstance(image, PImage.Image):
            image = np.array(image)
        res = await asyncio.to_thread(self.client.recognize_faces, image)
        return self.unpack_response(res)
    # async def arecognize_faces(self, image: Union[Image, PImage.Image, np.ndarray]):
    #     if type(image) == Image:
    #         image = image.pil_img
    #     if isinstance(image, PImage.Image):
    #         image = np.array(image)
    #     res = await self.client.async_recognize_faces(image)
    #     return self.unpack_response(res)
    

    async def ocr_image(self, image: Union[Image, PImage.Image, np.ndarray]):
        if type(image) == Image:
            image = image.pil_img
        res = await asyncio.to_thread(self.client.ocr_image, image=image)
        return res
    

    def unpack_response(self, response):
        output = []
        if response['status'] == 'ok':
            for i, face_json in enumerate(response['data']):
                output.append(Face(**face_json))
            return output
        else:
            raise Exception(response['data'])
        

    async def get_background_mask(self, image: Union[Image, PImage.Image, np.ndarray]):
        if type(image) == Image:
            image = image.pil_img
        if isinstance(image, PImage.Image):
            image = np.array(image)
        res = await asyncio.to_thread(self.client.get_background_mask, image)
        # res = self.client.get_background_mask(image)
        return Image.from_numpy(res)
        # return Image.from_pil(res['no_bg_image']), Image.from_pil(res['mask'])
    
    async def remove_background(self, image: Union[Image, PImage.Image, np.ndarray]):
        if type(image) == Image:
            image = image.pil_img
        if isinstance(image, PImage.Image):
            image = np.array(image)
        res = await asyncio.to_thread(self.client.remove_background, image)
        # return res
        # res = self.client.get_background_mask(image)
        return Image.from_numpy(res)
        



class Face:

    def __init__(
            self,
            bbox,
            kps,
            det_score,
            landmark_3d_68,
            pose,
            landmark_2d_106,
            gender,
            age,
            embedding,
            sex,
            normed_embedding,            
        ) -> None:
        self.bbox = np.array(bbox)
        self.kps = kps
        self.det_score = det_score
        self.landmark_3d_68 = np.array(landmark_3d_68)
        self.pose = np.array(pose)
        self.landmark_2d_106 = np.array(landmark_2d_106)
        self.gender = gender
        self.age = age
        self.embedding = np.array(embedding)
        self.sex = sex
        self.normed_embedding=np.array(normed_embedding)


    def to_dict(self, label=None):
        return {
            'label': label,
            'face': {
                'box': self.bbox.tolist(),
                'box_xmin': float(self.bbox[0]),
                'box_ymin': float(self.bbox[1]),
                'box_width': float(self.bbox[2] - self.bbox[0]),
                'box_height': float(self.bbox[3] - self.bbox[1]),
                'landmark': self.landmark_2d_106.tolist(),
                'pose': self.pose.tolist(),
            },
            'gender': int(self.gender),                    
            'age': self.age,
            }