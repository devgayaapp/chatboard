from PIL import Image
import requests
from transformers import AutoTokenizer, Blip2Processor, Blip2ForConditionalGeneration, Blip2Model, AutoProcessor
import torch



class Image2TextBLIP2:

    def __init__(self) -> None:        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
        )  # doctest: +IGNORE_RESULT
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.auto_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model_embeddings = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

    def complete(self, image, prompt):
        inputs = self.processor(images=image.pil_img, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text


    def text_embeddings(self, text):
        inputs = self.tokenizer(text=[text], return_tensors="pt", padding=True)
        text_features = self.model_embeddings.get_text_features(**inputs)
        return text_features
        

    def image_embeddings(self, image):
        inputs = self.auto_processor(images=image.pil_img, return_tensors="pt")
        image_features = self.model_embeddings.get_image_features(**inputs)
        return image_features
    

    def qformer_embeddings(self, image):
        inputs = self.processor(images=image.pil_img, return_tensors="pt")
        qformer_outputs = self.model_embeddings.get_qformer_features(**inputs)
        return qformer_outputs




