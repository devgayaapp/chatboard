import tempfile
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from components.image.image import Image

from config import TEMP_DIR




# BG_COLOR = (192, 192, 192) # gray
BG_COLOR = (0, 0, 0) # gray
MASK_COLOR = (255, 255, 255) # white


class ImageSegmentor:



    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path='models/mediapipe/deeplabv3.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options,
                                            output_category_mask=True)        
        self.segmenter = vision.ImageSegmenter.create_from_options(options)



    def segment(self, image: Image):
        # with tempfile.NamedTemporaryFile(suffix=f'.{image.ext}', dir=TEMP_DIR) as img_file:
        with tempfile.NamedTemporaryFile(suffix=f'.png', dir=TEMP_DIR) as img_file:
            if image.pil_img.mode != 'RGB':
                image.pil_img = image.pil_img.convert('RGB')
            image.to_file(img_file.name)
            image = mp.Image.create_from_file(img_file.name)
        # image = image.np_arr
            # image = mp.Image.create_from_rgb_cv_mat(image.pil_img)
            # Retrieve the masks for the segmented image
            segmentation_result = self.segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            return segmentation_result

            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
            return Image.from_numpy(output_image)

        print(f'Segmentation mask of {name}:')
        # resize_and_show(output_image)



    def segment_and_mask(self, image: Image):
        # with tempfile.NamedTemporaryFile(suffix=f'.{image.ext}', dir=TEMP_DIR) as img_file:
        with tempfile.NamedTemporaryFile(suffix=f'.png', dir=TEMP_DIR) as img_file:
            image.to_file(img_file.name)
            image = mp.Image.create_from_file(img_file.name)
        # image = image.np_arr
            # image = mp.Image.create_from_rgb_cv_mat(image.pil_img)
            # Retrieve the masks for the segmented image
            segmentation_result = self.segmenter.segment(image)
            category_mask = segmentation_result.category_mask

            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)
            return Image.from_numpy(output_image)

    def confidence_mask(self, segmentation_result, idx):
        category_mask = segmentation_result.confidence_masks[idx]
        image_data = category_mask.numpy_view()
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR[0]
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR[0]

        # condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        condition = np.stack((category_mask.numpy_view(),), axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        return Image.from_numpy(output_image)
    

        
    def mask_condition(self, image: Image):
        segmentation_result = self.segment(image)
        condition = np.stack((segmentation_result.category_mask.numpy_view(),) * 3, axis=-1) > 0.1
        return condition

