import hashlib
from components.apis.openai_api import get_openai_embeddings
from components.image.image import Image
from glob import glob
import numpy as np
from components.text.embeddings import cos_distance
from config import ASSETS_DIR
from thefuzz import fuzz
from retry import retry

import pickle
import cv2


import redis
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.client import NEVER_DECODE

from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

# OpenAI API key

 
# Redis connection details





class ControlNetRedis:

    def __init__(self, redis_host=REDIS_HOST, redis_port=REDIS_PORT) -> None:
        redis_password = REDIS_PASSWORD
        self.conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)
        if self.conn.ping():
            print("Connected to Redis")


    def hset(self, name, mapping):
        return self.conn.hset(name=name, mapping=mapping)
    

    def add_control_image(self, image_det, idx=None):
        if idx is None:
            idx = hashlib.md5(image_det.caption.encode('utf-8')).hexdigest()
        self.hset(f'control_img:{idx}', {
            'caption': image_det.caption,
            'image': image_det.image.to_stream().getvalue(),
            'embedding': np.array(image_det.emb).astype(np.float32).tobytes(),
            'pose': image_det.pose_image.to_stream().getvalue(),
            'canny': image_det.canny_image.to_stream().getvalue(),
            'mask': image_det.mask.to_stream().getvalue(),
            'width': image_det.canny_image.size[0],
            'height': image_det.canny_image.size[1],
            'number_of_main_people': image_det.stats['number_of_main_people'],
            'number_of_people': image_det.stats['number_of_people'],
            'categories': ','.join([o['category'] for o in image_det.objects]),
            'filename': image_det.filename,
        })

    def add_control_record(self, key, record):
        # if idx is None:
            # idx = hashlib.md5(record['caption'].encode('utf-8')).hexdigest()
        self.hset(key, {
            'caption': record['caption'],
            'image': record['image'].to_stream().getvalue(),
            'embedding': record['embedding'].astype(np.float32).tobytes(),
            'pose': record['pose'].to_stream().getvalue(),
            'canny': record['canny'].to_stream().getvalue(),
            'mask': record['mask'].to_stream().getvalue(),
            'width': record['canny'].size[0],
            'height': record['canny'].size[1],
            'number_of_main_people': record['number_of_main_people'],
            'number_of_people': record['number_of_people'],
            'categories': ','.join([o for o in record['categories']]),
            'filename': record['filename'],
        })
    
    def hget(self, key, field):

        if 'embedding' in field or 'image' in field: 
            value = self.conn.execute_command('HGET', key, field, NEVER_DECODE=True)
            if value is None:
                return None
            return value
        else:
            return self.conn.hget(key, field)
        

    def hmget(self, key, fields):
        # return self.conn.hmget(key, fields)
        command_args = [key] + fields

        value = self.conn.execute_command('HMGET', *command_args, NEVER_DECODE=True)
        
        if value is None:
            return None
        rec = {}
        for f,v in zip(fields, value):
            if f == 'caption':
                rec[f] = v.decode('utf-8')
            elif f == 'embedding':
                rec[f] = np.frombuffer(v, dtype=np.float32)
            elif f == 'image':
                rec[f] = Image.from_bytes(v)
            elif f == 'canny':
                rec[f] = Image.from_bytes(v)
            elif f == 'pose':
                rec[f] = Image.from_bytes(v)
            elif f == 'mask':
                rec[f] = Image.from_bytes(v)
            elif f == 'categories':
                rec[f] = v.decode('utf-8').split(',')
            elif f == 'filename':
                rec[f] = v.decode('utf-8')
            else:                
                rec[f] = int(v.decode('utf-8')) if v is not None else None
        return rec
    

    def keys(self, pattern):
        return self.conn.keys(pattern)

    def create_index(self):
        SCHEMA = [
            TextField("caption"),
            TextField("image"),
            TextField("canny"),
            TextField("pose"),
            TextField("mask"),
            TextField("filename"),
            NumericField("number_of_main_people"),
            NumericField("number_of_people"),
            NumericField("width"),
            NumericField("height"),
            TagField("categories"),
            VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
        ]
        try:
            self.conn.ft("controlnet").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["control_img:"], index_type=IndexType.HASH))
        except Exception as e:
            print("Index already exists")


    def search_vectors(
            self, 
            query_vector, 
            min_num_of_people=0, 
            max_num_of_people='inf', 
            top_k=5
        ):
        # if max_num_of_people == 'inf':
        #     base_query = f"@number_of_main_people:[$min_num_of_people inf]=>[KNN {top_k} @embedding $vector AS vector_score]"
        # else:
        #     base_query = f"@number_of_main_people:[$min_num_of_people $max_num_of_people]=>[KNN {top_k} @embedding $vector AS vector_score]"
        base_query = f"@number_of_main_people:[{min_num_of_people} {max_num_of_people}]=>[KNN {top_k} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields("caption", "vector_score").sort_by("vector_score").dialect(2)
    
        try:
            results = self.conn.ft("controlnet").search(query, query_params={
                "vector": query_vector, 
                # 'min_num_of_people': min_num_of_people,
                # "max_num_of_people": max_num_of_people
            })
        except Exception as e:
            print("Error calling Redis search: ", e)
            raise e
    
        return results
    
    def search_vectors2(self, query_vector, top_k=5):
        base_query = f"*=>[KNN {top_k} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields("caption", "vector_score").sort_by("vector_score").dialect(2)
    
        try:
            results = self.conn.ft("controlnet").search(query, query_params={"vector": query_vector})
        except Exception as e:
            print("Error calling Redis search: ", e)
            return None
    
        return results
        
    
    
    
    def get_all(self, pattern="control_img:*", field=None):
        hash_keys = self.conn.keys(pattern)  # Change the pattern as needed

        # Fetch values for each hash key
        data = {}
        for key in hash_keys:
            print('k', key)
            data[key] = self.conn.hgetall(key, field, encoding=NEVER_DECODE)

        for key, value in data.items():
            print(key, value)
    
    def scan(self, pattern="*"):
        keys = []
        cursor = '0'
        # pattern = "*"  # Change the pattern as needed

        while cursor != 0:
            cursor, partial_keys = self.conn.scan(cursor, match=pattern)
            keys.extend(partial_keys)

        # Fetch values for each key
        data = {}
        for key in keys:
            data[key.decode()] = self.conn.hget(key).decode()

        # Print the data
        for key, value in data.items():
            print(key, value)


    def delete_all(self):
        self.conn.flushall()
        print("Deleted all keys")


    

    

def transform_to_background(image):
    if image is None:
        return None
    canny_image = image.np_arr
    # zero out middle columns of image where pose will be overlayed
    zero_start = canny_image.shape[1] // 4
    zero_end = zero_start + canny_image.shape[1] // 2
    canny_image[:, zero_start:zero_end, :] = 0

    # canny_image = canny_image[:, :, None]
    # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.from_numpy(canny_image)
    return canny_image



# def format2styles(subject, background):
#     if subject == 'none':
#         return None, transform_to_background(get_control_image(background)), get_image_params(background)
#     if subject == 'portrait':
#         return None, get_control_image('portrait'), get_image_params('portrait')
#     if subject == 'object':
#         return None, get_control_image('object'), get_image_params('object')
#     if subject == 'building':
#         return None, get_control_image('building'), get_image_params('street')
    
#     if subject == 'fullbody':
#         return get_control_image('pose'), transform_to_background(get_control_image(background)), get_image_params(background)
#     if subject == 'sport':
#         return get_control_image('sport'), transform_to_background(get_control_image(background)), get_image_params(background)
#     if subject == 'interaction':
#         return get_control_image('interaction'), transform_to_background(get_control_image(background)), get_image_params('documentary')
#     if subject == 'group':
#         return get_control_image('group'), transform_to_background(get_control_image(background)), get_image_params('documentary')


print('loading control images...', str(ASSETS_DIR))



def recrop_control_images(width, height, canny_image=None, pose_image=None):

    if canny_image is not None:
        canny_image = canny_image.resize(height=height)

        if canny_image.size[0] < width:
            canny_image = canny_image.resize(width=width)
            size_diff = np.abs((height - canny_image.size[1])) // 2
            canny_image = canny_image.crop(0, size_diff, width, height)
        elif canny_image.size[0] > width:
            size_diff = np.abs(canny_image.size[0] - width) // 2
            canny_image = canny_image.crop(size_diff, 0, width, height)
    else:
        canny_image = Image.plain_color(width, height, color=(0, 0, 0))
    
    if pose_image is not None:
        pose_image = pose_image.resize(height=height)
        if pose_image.size[0] > width:
            size_diff = np.abs(pose_image.size[0] - width) // 2
            pose_image = pose_image.crop(size_diff, 0, width, height)

    # pose_image = pose_image.resize(height=height)

        pose_image_arr = pose_image.np_arr.copy()
        canny_image_arr = canny_image.np_arr.copy()


        # zero out middle columns of image where pose will be overlayed
        zero_start = (canny_image_arr.shape[1] - pose_image_arr.shape[1]) // 2
        zero_end = zero_start + pose_image_arr.shape[1]
        canny_image_arr[:, zero_start:zero_end, :] = 0

        # canny_image = canny_image[:, :, None]
        # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        croped_canny_image = Image.from_numpy(canny_image_arr)
        

        dialated_pose_arr = np.zeros_like(canny_image_arr)
        dialated_pose_arr[:, zero_start:zero_end, :] = pose_image_arr
        dialated_pose_image = Image.from_numpy(dialated_pose_arr)
    else:
        croped_canny_image = canny_image
        dialated_pose_image = Image.plain_color(width, height, (0, 0, 0))

    return croped_canny_image, dialated_pose_image



def get_control_files_from_prompt(prompt, is_prompt_contains_people=True):
    emb = get_openai_embeddings(prompt)[0]

    # is_prompt_contains_people = does_the_prompt_contains_people(prompt)

    def get_control_image_idx(contorol_emb, emb):
        diff = cos_distance(contorol_emb, emb)
        #! getting the top match
        file_index = np.argmax(diff)
        #! add randomization
        top_indices = np.argsort(diff.flatten())[:10]

        # top_indices = np.argsort(diff.flatten())
        for i in list(top_indices):
            # print(diff[i], canny_file_list[i].split('/')[-1])
            # print(diff[i], background_captions[i])
            print(diff[i],  canny_file_list[i].split('/')[-1])
            print(diff[i], fuzz.ratio(prompt, background_captions[i]), background_captions[i])
        # file_index = np.random.choice(top_indices)
        print('distance', diff[file_index])
        return file_index
    


    print(prompt, 'contains people:', is_prompt_contains_people)
    
    # canny_index = get_control_image_idx(canny_emb, emb)
    # canny_file = canny_file_list[canny_index]
    # print('canny', canny_file.split('/')[-1])
    # # canny_image = Image.from_file(canny_file)
    # pose_image = None
    # pose_file = None
    # mask_file = None
    # data_file = None
    # if is_prompt_contains_people:
        
    #     pose_index = get_control_image_idx(pose_emb, emb)
    #     pose_file = pose_file_list[pose_index]
    #     print('pose:', pose_file.split('/')[-1])
    #     # pose_image = Image.from_file(pose_file)
    #     mask_file = pose_masks_list[pose_index]
    #     data_file = pose_data_list[pose_index]
        
        

    return canny_file, pose_file, mask_file, data_file


# def get_control_images(canny_file, pose_file):
#     canny_image = Image.from_file(canny_file)
#     pose_image = None
#     if pose_file:
#         pose_image = Image.from_file(pose_file)
#     return canny_image, pose_image








def find_closed_composition_id(control_net_redis, generic_prompt, min_num_of_people=0, max_num_of_people = 'inf', logger=None):
    def print_log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    print_log(f'generic prompt: {generic_prompt}')


    query_vector = get_openai_embeddings(generic_prompt)
    query_vector = np.array(query_vector).astype(np.float32).tobytes()

    results = control_net_redis.search_vectors(
        query_vector, 
        min_num_of_people=min_num_of_people, 
        max_num_of_people=max_num_of_people
    )


    if results:
        results_score = []
        for i, rec in enumerate(results.docs):
            score = 1 - float(rec.vector_score)
            similarity_score = round(score ,3)
            fuzz_score = round(fuzz.ratio(generic_prompt, rec.caption)/100, 3)
            final_score = round((similarity_score + fuzz_score)/2, 3)
            results_score.append({
                'id': rec.id,
                'caption': rec.caption,
                'similarity_score': similarity_score,
                'fuzz_score': fuzz_score,
                'final_score': final_score,
            })
        results_score = sorted(results_score, key=lambda x: x['final_score'], reverse=True)
        for i, rec in enumerate(results_score):
            print_log(f"\t{i}. {rec['id']} {rec['caption']} (Score:{rec['final_score']}  sim score:{rec['similarity_score']} fuzz: {rec['fuzz_score']})")

        if min_num_of_people > 0: #! return only pose
            return None, results_score[0]['id']
        else: #! return only canny
            return results_score[0]['id'], None
    else: 
        print_log('no results')
        return None
    

def fetch_control_images(control_net_redis, canny_image_id=None, pose_image_id=None, logger=None):
    canny_image = None
    pose_image = None
    if canny_image_id:
        record = control_net_redis.hmget(canny_image_id, ['image', 'caption', 'embedding', 'number_of_main_people', 'categories', 'pose', 'canny'])
        canny_image = record['canny']
    if pose_image_id:
        record = control_net_redis.hmget(pose_image_id, ['image', 'caption', 'embedding', 'number_of_main_people', 'categories', 'pose', 'canny'])
        pose_image = record['pose']
    return canny_image, pose_image



def round_to_divisible_by_eight(number):
    rounded_number = round(number / 8) * 8
    return rounded_number


def generate_control_images(controlnet_redis, prompt, min_num_of_people=0, max_num_of_people = 'inf', width=None, height=None, logger=None):
    canny_image_id, pose_image_id = find_closed_composition_id(
        controlnet_redis, 
        prompt, 
        min_num_of_people=min_num_of_people,
        max_num_of_people=max_num_of_people,
        logger=logger
        )

    canny_img, pose_image = fetch_control_images(controlnet_redis, canny_image_id, pose_image_id, logger=logger)
    if not height:
        height = 768

    if not width:
        if canny_img:
            width = int(canny_img.size[0] * height / canny_img.size[1])
        else:
            width = 768

    print(f"height: {height} width: {width}")
        # height = canny_img.size[1]
    height = round_to_divisible_by_eight(height)
    width = round_to_divisible_by_eight(width)

    print(f"post fix: height: {height} width: {width}")

    canny_image, pose_image  = recrop_control_images(width, height, canny_img, pose_image)
    return canny_image, pose_image