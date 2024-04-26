from controlnet_aux import OpenposeDetector
from components.image.image import Image



class PoseDetector:

    def __init__(self):
        self.model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")


    def detect(self, img: Image):
        pose_img = self.model(img.pil_img)
        return Image.from_pil(pose_img)
