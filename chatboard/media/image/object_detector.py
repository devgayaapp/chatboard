import tempfile
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from components.detection.cv_utils import visualize
from components.image.image import Image

from config import TEMP_DIR


class ObjectDetector:


    def __init__(self, score_threshold=0.5) -> None:
        base_options = python.BaseOptions(model_asset_path='models/mediapipe/efficientdet.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                            score_threshold=score_threshold)
        self.detector = vision.ObjectDetector.create_from_options(options)

    
    def detect(self, image: Image, annotate=False):
        # image = image.np_arr
        # with tempfile.NamedTemporaryFile(suffix=f'.{image.ext}', dir=TEMP_DIR) as img_file:
        with tempfile.NamedTemporaryFile(suffix=f'.png', dir=TEMP_DIR) as img_file:
            image.to_file(img_file.name)
            image = mp.Image.create_from_file(img_file.name)
            detection_result = self.detector.detect(image)
            # print('------------------------')
            # for detection in detection_result.detections:
            #     for cat in detection.categories:
            #         print(cat.category_name)

            image_copy = np.copy(image.numpy_view())
            if annotate:
                annotated_image = visualize(image_copy, detection_result)
                rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                annotated_image = Image.from_numpy(rgb_annotated_image)
                return detection_result, annotated_image
                    
            return detection_result
        
        