from typing import Union
from components.image.image import Image
import PIL as pil
try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    print("transformers not installed")
import torch
import numpy as np


model_id = "openai/clip-vit-base-patch32"


class Image2TextCLIP:


    def __init__(self) -> None:
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # move the model to the device
        self.model.to(self.device)


    def get_image_embeddings(self, image: Image, normalize: bool =True):
        if isinstance(image, Image):
            image = image.pil_img
        image = self.processor(
            text=None,
            images=image,
            return_tensors='pt'
        )['pixel_values'].to(self.device)        
        image_emb = self.model.get_image_features(image).detach().cpu().numpy()
        if normalize:
            image_emb = image_emb / np.linalg.norm(image_emb)
        return image_emb
    
    def get_text_embeddings(self, text: str, normalize: bool=True):
        label_tokens = self.processor(
            text=text,
            padding=True,
            images=None,
            return_tensors='pt'
        ).to(self.device)
        text_features = self.model.get_text_features(**label_tokens).detach().cpu().numpy()
        if normalize:
            text_features = text_features / np.linalg.norm(text_features)
        return text_features
    

