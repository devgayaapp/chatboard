
from IPython.display import display
from components.text.image_prompts import transform_to_generic_names
# from components.text.image_prompts import nlp_object_detections
from components.text.nlp.text_object_extractor import extract_text_objects

from components.text.nlp.style_spliter import nlp_style_spliter
from retry import retry

from components.text.util import sanitize_sentence
import pickle

from urllib.request import urlretrieve
from json import load
from components.image.image import Image
from os.path import join
import os
from IPython.display import display
import shutil


import components.image.image as image_utils
from glob import glob
from config import DATA_DIR
from IPython.display import display
from components.image.image_describer import ImageDescriber
from components.image.object_detector import ObjectDetector     
from components.image.image_segmentor import ImageSegmentor
from components.image.pose_detector import PoseDetector
from components.apis.openai_api import get_openai_embeddings
from components.detection.face_detection import detect_faces_image_sa

from components.image.canny import canny


from sklearn.cluster import KMeans
import numpy as np

def split_into_groups(sizes, num_groups):
    # Convert sizes to a numpy array
    sizes = np.array(sizes)

    # Reshape the sizes array to fit the K-means input format
    sizes_reshaped = sizes.reshape(-1, 1)

    # Initialize K-means clustering with the desired number of groups
    kmeans = KMeans(n_clusters=num_groups, random_state=0)

    # Fit the data to the K-means model
    kmeans.fit(sizes_reshaped)

    # Get the cluster labels for each individual
    labels = kmeans.labels_

    # Create an empty dictionary to store the individuals for each group
    groups = {}
    for i in range(num_groups):
        groups[i] = []

    # Assign each individual to the corresponding group based on the cluster labels
    for i, label in enumerate(labels):
        groups[label].append(i)

    return groups


@retry(delay=1, backoff=5, max_delay=60)
def extract_text_objects_safe(caption):
    return extract_text_objects(caption)



def detection_stats(detection_result, image, caption, verbose=False):
    objects = []
    im_width, im_height = image.size
    im_center_x = im_width / 2
    im_center_y = im_height / 2

    for detection in detection_result.detections:
        center_x = round(detection.bounding_box.origin_x + detection.bounding_box.width / 2)
        center_y = round(detection.bounding_box.origin_y + detection.bounding_box.height / 2)
        obj = {
            'category': detection.categories[0].category_name,
            'confidence': detection.categories[0].score,
            'size': round(detection.bounding_box.width * detection.bounding_box.height),
            'relative_size': round((detection.bounding_box.width * detection.bounding_box.height) / (im_width * im_height), 2),
            'bounding_box': detection.bounding_box,
            'center': {'x': center_x, 'y':center_y},        
            'center_distance': round(((center_x - im_center_x)**2 + (center_y - im_center_y)**2)**0.5),
            'all_categories': [cat for cat in detection.categories],
            'is_main': False,
        }
        objects.append(obj)


    person_list = [(i,o['relative_size']) for i,o in enumerate(objects) if o['category'] == 'person']

    if person_list:
        max_size = max(person_list, key=lambda x: x[1])
        min_size = min(person_list, key=lambda x: x[1])
        
        has_main = False
        if max_size[1] > 1.5 * min_size[1]:# may be need to compare with image size
            sizes = [p[1] for p in person_list]
            num_groups = 2
            result = split_into_groups(sizes, num_groups)
            group1, group2 = result.items()
            # for group_id, individuals in result.items():
                # print("Main", group_id + 1, ":", individuals)
            # for i in group1[1]:
            group1_objects = [objects[person_list[i][0]] for i in group1[1]]
            group2_objects = [objects[person_list[i][0]] for i in group2[1]]

            group1_mean_size = np.mean([o['relative_size'] for o in group1_objects])
            group2_mean_size = np.mean([o['relative_size'] for o in group2_objects])
            if group1_mean_size > group2_mean_size:
                for o in group1_objects:
                    o['is_main'] = True
            else:
                for o in group2_objects:
                    o['is_main'] = True
            has_main = True
        else:
            for i,o in enumerate(objects):
                if objects[i]['category'] == 'person':
                    objects[i]['is_main'] = True
            
                # print('main', objects[person_list[i][0]]['category'])
        global_pos = {
            'global_left' : min([o['bounding_box'].origin_x for o in objects if o['category'] == 'person' and o['is_main'] == True]),
            'global_right' : max([o['bounding_box'].origin_x + o['bounding_box'].width for o in objects if o['category'] == 'person' and o['is_main'] == True]),
            'global_top' : min([o['bounding_box'].origin_y for o in objects if o['category'] == 'person' and o['is_main'] == True]),
            'global_bottom' : max([o['bounding_box'].origin_y + o['bounding_box'].height for o in objects if o['category'] == 'person' and o['is_main'] == True]),
        }
        global_pos['global_width'] = global_pos['global_right'] - global_pos['global_left']
        global_pos['global_height'] = global_pos['global_bottom'] - global_pos['global_top']
        # global_pos = {
        #     'global_left' : min([o['bounding_box'].origin_x for o in objects if o['category'] == 'person' and o['is_main'] == True]),
        #     'global_width' : max([o['bounding_box'].width for o in objects if o['category'] == 'person' and o['is_main'] == True]),
        #     'global_top' : min([o['bounding_box'].origin_y for o in objects if o['category'] == 'person' and o['is_main'] == True]),
        #     'global_height' : max([o['bounding_box'].height for o in objects if o['category'] == 'person' and o['is_main'] == True]),
        # }

    else:
        global_pos = {
            'global_left' : 0,
            'global_right' : image.size[0],
            'global_width' : image.size[0],
            'global_top' : 0,
            'global_bottom': image.size[1],
            'global_height' : image.size[1],
        }
        # for i,o in enumerate(objects):
        #     if objects[i]['category'] == 'person':
        #         objects[i]['is_main'] = True
        #         print('main', objects[i]['category'])

    number_of_main_people = len([o for o in objects if o['is_main'] == True])

    number_of_animals = 0

    size_mean = np.mean([o['relative_size'] for o in objects])
    size_var = np.var([o['relative_size'] for o in objects])
    size_std = np.mean([o['relative_size'] for o in objects])

    text_objects = extract_text_objects_safe(caption)

    number_of_people = len([o for o in objects if o['category'] == 'person'])
    
    if text_objects['people'] > number_of_people:
        number_of_main_people = text_objects['people']

    if text_objects['animals'] > 0:
        number_of_animals = text_objects['animals']
    
        # [np.abs(o['relative_size']) - size_mean for o in objects]
        # print(detection.bounding_box)
        # print('size', detection.bounding_box.width * detection.bounding_box.height)
        # for cat in detection.categories:
            # print(cat)
            # print(cat.category_name)
    stats = {
        'size_mean': size_mean, 
        'size_var': size_var, 
        'size_std': size_std, 
        'number_of_main_people': number_of_main_people,
        'number_of_people': len([o for o in objects if o['category'] == 'person']),
        'number_of_animals': number_of_animals
    }
    stats.update(global_pos)
    return objects, stats

class DetectionImage:

    def __init__(
            self, 
            filename, 
            image, 
            croped_image, 
            mask, 
            annot_img, 
            canny_image, 
            pose_image, 
            caption, 
            emb, 
            detection_result, 
            segmentation_result, 
            faces, 
            face_image, 
            objects, 
            stats 
            ):
        self.image = image
        self.croped_image = croped_image
        self.mask = mask
        self.annot_img = annot_img
        self.canny_image = canny_image
        self.pose_image = pose_image
        self.caption = caption
        self.emb = emb
        self.detection_result = detection_result
        self.segmentation_result = segmentation_result
        self.faces = faces
        self.face_image = face_image
        self.objects = objects
        self.stats = stats
        self.filename = filename
        self.is_pose_validated = check_if_person_exists(self.pose_image)

    @property
    def number_of_people(self):
        if not self.is_pose_validated:
            return 0
        return self.stats['number_of_people']
    
    @property
    def number_of_main_people(self):
        if not self.is_pose_validated:
            return 0
        return self.stats['number_of_main_people']

    @property
    def number_of_animals(self):
        return self.stats['number_of_animals']

        

    def print_image_stats(self, show_image=False, show_all_images=False):
        objects, stats = detection_stats(self.detection_result, self.image)
        categories = [(o['category'], o['relative_size']) for o in objects]
        print(self.caption)
        print(f'mean', stats['size_mean'], 'std', stats['size_std'], 'size_var', stats['size_var'])
        print('main people:', stats['number_of_main_people'])
        print(categories)
        if show_image:
            display(self.image)
        if show_all_images:
            display(image_utils.Image.grid([self.image, self.mask, self.annot_img, self.canny_image, self.pose_image, self.face_image]))


    def get_name(self):
        return sanitize_sentence(self.caption).replace(' ', '_')

    def save(self, ctl_img_type, idx):
        if idx is not None:
            filename = f"{idx}_{self.get_name()}.png"
        else:
            filename = f"{self.get_name()}.png"
        if ctl_img_type == 'canny':
            self.canny_image.to_file(str(DATA_DIR /'controlnet'/ 'canny_images' / filename))
        elif ctl_img_type == 'pose':
            self.pose_image.to_file(str(DATA_DIR /'controlnet'/ 'pose_images' / filename))
            self.mask.to_file(str(DATA_DIR /'controlnet'/ 'pose_masks' / filename))
            with open(str(DATA_DIR /'controlnet'/ 'pose_data' / filename), 'wb') as f:
                pickle.dump({
                    'objects': self.objects,
                    'stats': self.stats,
                }, f)
        else:
            raise Exception('unknown ctl_img_type')


def np_to_cv2(np_image):
    open_cv_image = np_image[:, :, ::-1].copy() 
    return open_cv_image


def check_if_person_exists(pose_image):
    res = sum(sum(sum(pose_image.np_arr != 0)))
    if res > 100:
        return True
    return False





class ControlImageProcessor():


    def __init__(self):  
        print('initializing classifers...')      
        self.img_desc = ImageDescriber()
        print('image describer initialized')
        self.img_seg = ImageSegmentor()
        print('image segmentor initialized')
        self.obj_det = ObjectDetector()
        print('object detector initialized')
        self.pose_detector = PoseDetector()
        print('pose detector initialized')
        print('all classifiers initialized')
        # self.low_thresh_obj_det = ObjectDetector(score_thrxeshold=0.1)
        # print('low threshold object detector initialized')


    def process_image(self, image, prompt, default_height=512, num_beams=10, max_length=20, min_length=6):        
        image = image.resize(height=default_height)

        caption = self.img_desc.describe(image, max_length=max_length, min_length=min_length, num_beams=num_beams)[0]
        
        detection_result, annot_img = self.obj_det.detect(image, annotate=True)
        objects, stats = detection_stats(detection_result, image, prompt)

        croped_image = image.crop(stats['global_left'], 0, stats['global_width'], default_height)
        segmentation_result = self.img_seg.segment(croped_image)

        pose_image = self.pose_detector.detect(croped_image)
        pose_image = pose_image.resize(height=default_height)

        mask = self.img_seg.segment_and_mask(croped_image)
        mask = mask.resize(height=default_height)

        canny_image = canny(croped_image)
        canny_image = canny_image.resize(height=default_height)

        faces, face_image = detect_faces_image_sa(np_to_cv2(croped_image.np_arr), draw=True)
        face_image = image_utils.Image.from_numpy(face_image)

        # emb = get_openai_embeddings(caption)

        return DetectionImage(
            filename=None, 
            image=image, 
            croped_image=croped_image, 
            mask=mask, 
            annot_img=annot_img, 
            canny_image=canny_image, 
            pose_image=pose_image, 
            caption=caption, 
            emb=None, 
            detection_result=detection_result, 
            segmentation_result=segmentation_result, 
            faces=faces, 
            face_image=face_image, 
            objects=objects, 
            stats=stats)


    # def process_image_local(self, f, prompt=None, default_height=512, num_beams=10, max_length=20, min_length=6):
    #     if 'http' in f:
    #         image = image_utils.Image.from_url(f)
    #     else:
    #         image = image_utils.Image.from_file(f)
    #     # image = image.resize(height=768)
    #     image = image.resize(height=default_height)
    #     if prompt is None:
    #         caption = self.img_desc.describe(image, max_length=max_length, min_length=min_length, num_beams=num_beams)[0]
    #     else:
    #         caption = prompt

    #     caption = self.img_desc.describe(image, max_length=max_length, min_length=min_length, num_beams=num_beams)[0]

    #     detection_result, annot_img = self.obj_det.detect(image, annotate=True)
    #     objects, stats = detection_stats(detection_result, image, caption)

    #     croped_image = image.crop(stats['global_left'], 0, stats['global_width'], default_height)
    #     segmentation_result = self.img_seg.segment(croped_image)

    #     pose_image = self.pose_detector.detect(croped_image)
    #     pose_image = pose_image.resize(height=default_height)

    #     # if len(objects) == 0 and check_if_person_exists(pose_image):
    #     #     get_objects_from_gpt_process(caption)            
    #     mask = self.img_seg.segment_and_mask(croped_image)
    #     mask = mask.resize(height=default_height)

    #     canny_image = canny(croped_image)
    #     canny_image = canny_image.resize(height=default_height)

    #     faces, face_image = detect_faces_image_sa(np_to_cv2(croped_image.np_arr), draw=True)
    #     face_image = image_utils.Image.from_numpy(face_image)
    #     # print(caption)
    #     emb = get_openai_embeddings(caption)
    #     # print(len(emb[0]))  
    #     return DetectionImage( 
    #         filename= f,
    #         image= image,
    #         croped_image= croped_image,
    #         mask= mask,
    #         annot_img= annot_img,
    #         canny_image= canny_image,
    #         pose_image= pose_image,
    #         caption= caption,
    #         emb= emb[0],
    #         detection_result= detection_result,
    #         segmentation_result= segmentation_result,
    #         faces= faces,
    #         face_image= face_image,
    #         objects= objects,
    #         stats= stats,
    #     )
    




class DiffusionDBImages:

    def __init__(self):
        pass

    def download_part(self, part_id):
        part_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/images/part-{part_id:06}.zip'
        urlretrieve(part_url, f'tmp/part-{part_id:06}.zip')
        shutil.unpack_archive(f'tmp/part-{part_id:06}.zip', f'tmp/part-{part_id:06}')
        os.remove(f'tmp/part-{part_id:06}.zip')

    def get_image(self, image_name, part_id):
        if not os.path.isdir(f'tmp/part-{int(part_id):06}'):
            self.download_part(part_id)
        return Image.from_file(join(f'tmp/part-{int(part_id):06}', image_name))