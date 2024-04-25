from typing import List
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image



# max_length = 100
# num_beams = 10

class ImageDescriber:

    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', self.device)
        # model.to(device)
        self.model.to(self.device)
        print('loaded image describer')


    def describe(self, images, max_length=16, min_length=1, num_beams=4):
        if type(images) != list:
            images = [images]
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "min_length": min_length}
        image_list = []
        for im in images:
            # i_image = Image.open(image_path)
            i_image = im.pil_img
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            image_list.append(i_image)

        pixel_values = self.feature_extractor(images=image_list, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds




    def predict_step_files(self, image_paths, max_length=16, min_length=1, num_beams=4):
        if type(image_paths) == str:
            image_paths = [image_paths]
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "min_length": min_length}
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


