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
from components.text.nlp.text_object_extractor import extract_text_objects


from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD
from util.logger import get_logger





def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()
    
def hash_image(rec):
    tags = ['prompt', 'cfg', 'steps', 'seed', 'width', 'height', 'sampler', 'image_nsfw']
    image_str = '_'.join([str(rec[t]) for t in tags])
    return hashlib.md5(image_str.encode()).hexdigest()



@retry(delay=1, backoff=5, max_delay=60)
def extract_text_objects_safe(caption, logger=None):
    return extract_text_objects(caption, logger=logger)


def get_search_params(prompt, logger=None):
    objects = extract_text_objects_safe(prompt, logger=logger)
    object_tags = [o['text'] for o in objects.objects['object']]
    animal_tags = [o['text'] for o in objects.objects['animal']]
    people_tags = [o['text'] for o in objects.objects['humanoid']]
    place_tags = [o['text'] for o in objects.objects['place']]

    tags = object_tags + animal_tags + people_tags + place_tags

    if objects['people'] >= 3:
        return 3, 'inf', tags
    else:
        return objects['people'], objects['people'], tags

class PromptResult:
    pass

class ImageResult:
    pass

class ContolNetPromptRedis:

    def __init__(self, redis_host=REDIS_HOST, redis_port=REDIS_PORT) -> None:
        redis_password = REDIS_PASSWORD
        self.conn = redis.Redis(host=redis_host, port=redis_port, password=redis_password, encoding='utf-8', decode_responses=True)        
        if self.conn.ping():
            print("Connected to Redis")


    def hset(self, name, mapping):
        return self.conn.hset(name=name, mapping=mapping)
    
    def hmset(self, name, mapping):
        return self.conn.hmset(name=name, mapping=mapping)


    def add_prompt(
        self,
        prompt_id: str,
        prompt: str,
        style_prompt: str = None,
        embedding: np.ndarray = None,
        number_of_main_people: int = None,
        number_of_people: int = None,
        number_of_animals: int = None,
        categories: list[str] = None,
        prompt_nsfw: float = None,
        part_id: str=None,
        is_processed: int = None,
    ):
        # prompt_id = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        if not 'control_prompt:' in prompt_id:
            prompt_id = f'control_prompt:{prompt_id}'
        params = {}
        if prompt is not None:
            params['prompt'] = prompt
        if style_prompt is not None:
            params['style_prompt'] = style_prompt
        if embedding is not None:
            params['embedding'] = np.array(embedding).astype(np.float32).tobytes()
        if number_of_main_people is not None:
            params['number_of_main_people'] = number_of_main_people
        if number_of_people is not None:
            params['number_of_people'] = number_of_people
        if number_of_animals is not None:
            params['number_of_animals'] = number_of_animals
        if categories is not None:
            params['categories'] = ','.join(categories)
        if prompt_nsfw is not None:
            params['prompt_nsfw'] = prompt_nsfw
        if part_id is not None:
            params['part_id'] = part_id
        if is_processed is not None:
            params['is_processed'] = is_processed
        self.hset(prompt_id, params)

    def update_prompt(
        self,
        prompt_id: str,
        prompt: str = None,
        style_prompt: str = None,
        embedding: np.ndarray = None,
        number_of_main_people: int = None,
        number_of_people: int = None,
        number_of_animals: int = None,
        categories: list[str] = None,
        prompt_nsfw: float = None,
        images: list[str] = None,
        part_id: str=None,
        is_processed: int = None,
    ):
        params = {}
        if prompt is not None:
            params['prompt'] = prompt
        if style_prompt is not None:
            params['style_prompt'] = style_prompt
        if embedding is not None:
            params['embedding'] = np.array(embedding).astype(np.float32).tobytes()
        if number_of_main_people is not None:
            params['number_of_main_people'] = number_of_main_people
        if number_of_people is not None:
            params['number_of_people'] = number_of_people
        if number_of_animals is not None:
            params['number_of_animals'] = number_of_animals
        if categories is not None:
            params['categories'] = categories
        if prompt_nsfw is not None:
            params['prompt_nsfw'] = prompt_nsfw
        if images is not None:
            params['images'] = images
        if part_id is not None:
            params['part_id'] = part_id
        if is_processed is not None:
            params['is_processed'] = is_processed
        self.hmset(f'control_prompt:{prompt_id}', params)

    def get_prompt(
        self,
        key: str,
        fields: list[str] = None
        # prompt: str,
    ):
        if 'control_prompt:' not in key:
            key = f'control_prompt:{key}'
        if fields is None:
            fields = ['prompt', 'style_prompt', 'embedding', 'number_of_main_people', 'number_of_animals', 'number_of_people', 'prompt_nsfw', 'categories', 'images', 'is_processed', 'part_id']
        command_args = [key] + fields
        value = self.conn.execute_command('HMGET', *command_args, NEVER_DECODE=True)

        if value is None:
            return None
        

        rec = PromptResult()
        for f,v in zip(fields, value):
            if v is None:
                continue
            if f in ['embedding']:
                setattr(rec, f, np.frombuffer(v, dtype=np.float32))
            elif f in ['categories']:
                setattr(rec, f, v.decode('utf-8').split(','))
            elif f in ['prompt', 'style_prompt']:
                # rec[f] = v.decode('utf-8')
                setattr(rec, f, v.decode('utf-8'))
            elif f in ['number_of_main_people', 'number_of_people', 'is_processed', 'part_id']:
                # rec[f] = int(v.decode('utf-8'))
                setattr(rec, f, int(v.decode('utf-8')))
            elif f in ['prompt_nsfw']:
                # rec[f] = float(v.decode('utf-8'))
                setattr(rec, f, float(v.decode('utf-8')))
            elif f in ['images']:
                # rec[f] = v.decode('utf-8').split(',')
                setattr(rec, f, v.decode('utf-8').split(','))
        # rec = {}
        # for f,v in zip(fields, value):
        #     if f in ['embedding']:
        #         rec[f] = np.frombuffer(v, dtype=np.float32)
        #     elif f in ['categories']:
        #         rec[f] = v.decode('utf-8').split(',')
        #     elif f in ['prompt', 'style_prompt']:
        #         rec[f] = v.decode('utf-8')
        #     elif f in ['number_of_main_people', 'number_of_people']:
        #         rec[f] = int(v.decode('utf-8'))
        #     elif f in ['prompt_nsfw']:
        #         rec[f] = float(v.decode('utf-8'))
        #     elif f in ['images']:
        #         rec[f] = v.decode('utf-8').split(',')
        setattr(rec, 'id', key)
        return rec
    

    def search_prompt(self, 
            query_vector, 
            min_num_of_people=0, 
            max_num_of_people='inf', 
            tags=None,
            top_k=5):
        
        if type(query_vector) == list:
            query_vector = np.array(query_vector).astype(np.float32).tobytes()
        elif type(query_vector) == np.ndarray:
            query_vector = query_vector.astype(np.float32).tobytes()
        
        base_query = f"(@number_of_main_people:[{min_num_of_people} {max_num_of_people}] @is_processed:[1 1]) =>[KNN {top_k} @embedding $vector AS vector_score]"
        # base_query = f"*=>[KNN {top_k} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields("prompt", "style_prompt", "prompt_nsfw", "images", "vector_score", "number_of_main_people", "number_of_people", "number_of_animals").sort_by("vector_score").dialect(2)
        # query = Query(base_query).return_fields("prompt", "vector_score").sort_by("vector_score").dialect(2)
    
        try:
            # results = self.conn.ft("controlnet").search(query, query_params={
            results = self.conn.ft("idx:control_prompt").search(query, query_params={
                "vector": query_vector, 
                # 'min_num_of_people': min_num_of_people,
                # "max_num_of_people": max_num_of_people
            })
            for res in results.docs:
                if hasattr(res, 'images'):
                    res.images = res.images.split(',')
                if hasattr(res, 'number_of_main_people'):
                    res.number_of_main_people = int(res.number_of_main_people)
                if hasattr(res, 'number_of_people'):
                    res.number_of_people = int(res.number_of_people)
                if hasattr(res, 'number_of_animals'):
                    res.number_of_animals = int(res.number_of_animals)
                if hasattr(res, 'prompt_nsfw'):
                    res.prompt_nsfw = float(res.prompt_nsfw)
                
        except Exception as e:
            print("Error calling Redis search: ", e)
            raise e
    
        return results.docs
    
    def add_images_to_prompt(
        self,
        prompt_id: str,
        image_ids: list[str],
    ):
        self.hmset(f'control_prompt:{prompt_id}', {
            'images': ','.join(image_ids)
        })
        # self.conn.sadd(f'control_prompt:{prompt_id}:images', *image_ids)


    def index_prompt(self):
        SCHEMA = [
            TextField('prompt'),
            TextField('style_prompt'),
            # VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
            VectorField('embedding', 'FLAT', {'TYPE': 'FLOAT32', 'DIM': 1536, 'DISTANCE_METRIC': 'COSINE'}),
            # VectorField("embedding", "FLAT", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
            NumericField('number_of_main_people'),
            NumericField('number_of_people'),
            NumericField('number_of_animals'),
            NumericField('prompt_nsfw'),
            NumericField('is_processed'),
            TagField('categories'),            
        ]
        try:
            self.conn.ft("idx:control_prompt").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["control_prompt:"], index_type=IndexType.HASH))
        except Exception as e:
            print("Index already exists")


    def drop_index_prompt(self):
        try:
            self.conn.ft("idx:control_prompt").dropindex()
        except Exception as e:
            print(e)

    def add_image(
        self,
        prompt_id: str,
        image_id: str,
        image_name: str,
        pose_image: Image=None,
        mask_image: Image=None,
        canny_image: Image=None,
        width: int=None,
        height: int=None,
        cfg: float=None,
        steps: int=None,
        seed: int=None,
        sampler: str=None,
        image_nsfw: float=None,
        is_processed: int=None,
        part_id: int=None,
        index: int=None,
        caption: str=None,
    ):
        if not 'control_image:' in image_id:
            image_id = f'control_image:{image_id}'
        params = {
            'prompt_id': prompt_id,
            'image_name': image_name,
            'width': width,
            'height': height,
            'cfg': cfg,
            'steps': steps,
            'seed': seed,
            'sampler': sampler,
            'image_nsfw': image_nsfw,
            'is_processed': is_processed,
            'index': index,
            'part_id': part_id,
            # 'caption': caption,
        }
        if pose_image:
            params['pose_image'] = pose_image.to_stream().getvalue()
        if mask_image:
            params['mask_image'] = mask_image.to_stream().getvalue()
        if canny_image:
            params['canny_image'] = canny_image.to_stream().getvalue()

        self.hset(image_id, params)

    def update_last_index(self, index):
        self.hset(f'last_index:0', {'index': index})

    def get_last_index(self):
        res = self.conn.hgetall(f'last_index:0')
        if res:
            return int(res['index'])
        raise Exception('No last index found')
    
    def update_image(
        self,
        image_id: str,
        pose_image: Image=None,
        mask_image: Image=None,
        canny_image: Image=None,
        caption: str=None,
        is_processed: int=None,
    ):
        params = {}
        if 'control_image' not in image_id:
            image_id = f'control_image:{image_id}'
        if pose_image:
            params['pose_image'] = pose_image.to_stream().getvalue()
        if mask_image:
            params['mask_image'] = mask_image.to_stream().getvalue()
        if canny_image:
            params['canny_image'] = canny_image.to_stream().getvalue()
        if is_processed is not None:
            params['is_processed'] = is_processed
        if caption is not None:
            params['caption'] = caption
        self.hset(image_id, params)


    def get_image(self, key, fields=None):
        if 'control_image:' not in key:
            key = f'control_image:{key}'
        if fields is None:
            fields = ['prompt_id', 'image_name', 'pose_image', 'mask_image', 'canny_image', 'width', 'height', 'cfg', 'steps', 'seed', 'sampler', 'image_nsfw', 'is_processed', 'part_id', 'index']
        command_args = [key] + fields
        value = self.conn.execute_command('HMGET', *command_args, NEVER_DECODE=True)

        if value is None:
            return None
        
        # rec = {}
        # for f,v in zip(fields, value):
        #     if f in ['pose_image', 'mask_image', 'canny_image']:
        #         im = Image.from_bytes(v)
        #         rec[f] = im
        #     elif f in ['width', 'height', 'steps', 'seed']:
        #         rec[f] = int(v.decode('utf-8'))
        #     elif f in ['cfg', 'image_nsfw']:
        #         rec[f] = float(v.decode('utf-8'))
        rec = ImageResult
        setattr(rec, 'id', key)
        for f,v in zip(fields, value):
            if f in ['pose_image', 'mask_image', 'canny_image']:
                im = None
                if not v is None:
                    im = Image.from_bytes(v)
                # rec[f] = im
                setattr(rec, f, im)
            elif f in ['sampler', 'prompt_id', 'image_name']:
                setattr(rec, f, v.decode('utf-8'))
            elif f in ['width', 'height', 'steps', 'seed', 'part_id', 'index', 'is_processed']:
                # rec[f] = int(v.decode('utf-8'))
                setattr(rec, f, int(v.decode('utf-8')))
            elif f in ['cfg', 'image_nsfw']:
                # rec[f] = float(v.decode('utf-8'))
                setattr(rec, f, float(v.decode('utf-8')))
        return rec
    

    def get_prompt_images(self, prompt_id):
        if 'control_prompt:' in prompt_id:
            split = prompt_id.split(':')
            prompt_id = split[1]
        base_query = f"@prompt_id:{prompt_id} @is_processed:[1 1]"
        query = Query(base_query).return_fields("prompt_id", "is_processed", "part_id", "index", "image_name", "width", "height", "cfg", "steps", "seed", "sampler", "image_nsfw")
        res = self.conn.ft("idx:control_image").search(query)
        random_image = np.random.choice(res.docs, 1)
        # image = self.get_image(random_image[0]['id'])
        image = self.get_image(random_image[0].id)
        return image

        
    def get_members(self, key):
        return self.conn.smembers(key)
    
    def get_image_keys(self):
        return self.get_members('control_image')

    def search_unprocessed_images(self, part_id=None):
        # query = f"@prompt_id:{{{prompt_id}}} @part_id:{{{part_id}}} @is_processed:0"
        if part_id is None:
            # base_query = f"'@is_processed:[0 0]' LIMIT 0 {limit}"
            base_query = f"@is_processed:[0 0]"
        else:
            base_query = f"'@is_processed:[0 0] @part_id:[{part_id} {part_id}]'"

        query = Query(base_query).return_fields("prompt_id", "is_processed", "part_id", "index", "image_name")
        res = self.conn.ft("idx:control_image").search(query)

        return res
    
    def index_image(self):
        SCHEMA = [
            TextField('prompt_id'),
            TextField('image_name'),
            TextField('pose_image'),
            TextField('mask_image'),
            TextField('canny_image'),
            TextField('caption'),
            NumericField('width'),
            NumericField('height'),
            NumericField('cfg'),
            NumericField('steps'),
            NumericField('seed'),
            TextField('sampler'),
            NumericField('image_nsfw'),
            NumericField('is_processed'),
            NumericField('part_id'),
            NumericField('index'),
        ]
        try:
            self.conn.ft("idx:control_image").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["control_image:"], index_type=IndexType.HASH))
            
        except Exception as e:
            print("Index already exists")

    def drop_index_image(self):
        try:
            self.conn.ft("idx:control_image").dropindex()
        except Exception as e:
            print(e)


    def reset_processed(self):
        # keys = self.get_image_keys()
        keys_image = self.conn.keys("control_image:*")
        keys_prompt = self.conn.keys("control_prompt:*")
        keys = keys_image + keys_prompt
        print(len(keys))
        for key in keys:
            self.hset(key, {'is_processed': 0})
        # print("Reset all processed images")


    def delete_all(self):
        self.conn.flushall()
        print("Deleted all keys")


def sigmoid_function(x, a=1, b=0):
    return 1 / (1 + np.exp(-x*a + b))


def calc_tag_score(text, tags, treshold=70):
    text = text.lower()
    text_words = text.split(' ')
    score = 0
    for word in text_words:
        for tag in tags:
            if fuzz.ratio(word, tag) > treshold:
                score += 1
    if len(tags) == 0:
        return 0
    return score / len(tags)
        

def linear_function(x, a=1, b=0):
    return x*a + b

class ControlnetImages:

    def __init__(self, logger=None) -> None:
        self.control_store = ContolNetPromptRedis()
        self.logger = logger


    def similar_prompts(self, prompt, num_of_people=None, fuzz_weight=0.1, tag_weight=0.4):
        vector_weight = 1 - fuzz_weight - tag_weight
        if num_of_people is None:
            min_num_people, max_num_people, tags = get_search_params(prompt, self.logger)
        else:
            min_num_people = num_of_people
            max_num_people = num_of_people
            tags = []
        prompt_emb = get_openai_embeddings(prompt)
        self.print_log(f"Searching for similar prompts (people: [{min_num_people}, {max_num_people}]) for: '{prompt}'")
        prompt_emb = prompt_emb[0]
        results_score = []
        results = self.control_store.search_prompt(prompt_emb, min_num_people, max_num_people, tags)
        for i, rec in enumerate(results):
            score = (1 - float(rec.vector_score))
            similarity_score = round(score ,3)
            fuzz_score = round(fuzz.ratio(prompt, rec.prompt)/100, 3)
            tag_score = round(calc_tag_score(rec.prompt, tags), 3)
            final_score = round((similarity_score * vector_weight + fuzz_score * fuzz_weight + tag_score * tag_weight), 3)            
            results_score.append({
                'id': rec.id,
                'prompt': rec.prompt,
                'style_prompt': rec.style_prompt,
                'similarity_score': similarity_score,
                'tag_score': tag_score,
                'fuzz_score': fuzz_score,
                'final_score': final_score,
                'number_of_main_people': rec.number_of_main_people,
                'number_of_people': rec.number_of_people,
                'number_of_animals': rec.number_of_animals,
            })
        results_score = sorted(results_score, key=lambda x: x['final_score'], reverse=True)
        
        for i, rec in enumerate(results_score):
            self.print_log(f"\t{i}. {rec['id']} {rec['prompt']} (Score:{rec['final_score']}  sim score:{rec['similarity_score']} fuzz: {rec['fuzz_score']})")

        top_sim_prompt = results_score[0]

        top_sim_prompt['search_params'] = {
            'min_num_people': min_num_people,
            'max_num_people': max_num_people,
        }

        return top_sim_prompt
    

    def prompt_images(self, prompt_id):
        result = self.control_store.get_prompt_images(prompt_id)
        self.print_log(f"control image: {result.id}")
        return result
    

    def get_control_images(self, prompt, use_controlnet_distance=True, width=None, height=None, num_of_people=None):
        top_sim_prompt = self.similar_prompts(prompt, num_of_people=num_of_people)
        ctr_images = self.prompt_images(top_sim_prompt['id'])
        top_sim_prompt['control_image'] = ctr_images.id
        height = height or ctr_images.height
        width = width or ctr_images.width
        if top_sim_prompt['number_of_main_people'] > 0:
            canny_image, pose_image = self.recrop_control_images(width, height, None, ctr_images.pose_image)
        else:
            canny_image, pose_image = self.recrop_control_images(width, height, ctr_images.canny_image, None)
        ctr_images.prompt = top_sim_prompt['prompt']
        ctr_images.style_prompt = top_sim_prompt['style_prompt']
        ctr_images.canny_image = canny_image
        ctr_images.pose_image = pose_image

        setattr(ctr_images, 'prompt_sim_data', top_sim_prompt)


        controlnet_conditioning_scale = [
            linear_function(top_sim_prompt['final_score'], a=1.666, b=-0.666),
            linear_function(top_sim_prompt['final_score'], a=1.666, b=-0.666) * 0.8,
            # top_sim_prompt['final_score'],
            # top_sim_prompt['final_score'] * 0.8,
            # sigmoid_function(top_sim_prompt['final_score'], a=30, b=20),
            # sigmoid_function(top_sim_prompt['final_score'], a=30, b=20)
        ] if use_controlnet_distance else None
        
        setattr(ctr_images, 'controlnet_conditioning_scale', controlnet_conditioning_scale)
        return ctr_images


    def print_log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)


    def recrop_control_images(self, width, height, canny_image=None, pose_image=None):

        
        if pose_image is not None:
            pose_image_resized = pose_image.resize(height=height)
            if pose_image_resized.size[0] < width:
                left_pad = (width - pose_image_resized.size[0]) // 2
                right_pad = width - pose_image_resized.size[0] - left_pad
                np_arr = np.full((height, width, 3), np.array([0,0,0]), dtype=np.uint8)
                np_arr[:, left_pad:width-right_pad, :] = pose_image_resized.np_arr
                pose_image_resized = Image.from_numpy(np_arr)
            elif pose_image_resized.size[0] > width:
                left_pad = (pose_image_resized.size[0] - width) // 2
                pose_image_resized = pose_image_resized.crop(left_pad, 0, width, height)
        else:
            pose_image_resized = Image.plain_color(width, height, color=(0, 0, 0))


        if canny_image is not None:
            canny_image_resized = canny_image.resize(height=height)
            if canny_image_resized.size[0] < width:
                left_pad = (width - canny_image_resized.size[0]) // 2
                right_pad = width - canny_image_resized.size[0] - left_pad
                np_arr = np.full((height, width, 3), np.array([0,0,0]), dtype=np.uint8)
                np_arr[:, left_pad:width-right_pad, :] = canny_image_resized.np_arr
                canny_image_resized = Image.from_numpy(np_arr)
            elif canny_image_resized.size[0] > width:
                left_pad = (canny_image_resized.size[0] - width) // 2
                canny_image_resized = canny_image_resized.crop(left_pad, 0, width, height)
            
            if pose_image is not None:
                canny_image_arr = canny_image_resized.np_arr
                pose_image_arr = pose_image_resized.np_arr

                zero_start = (canny_image_arr.shape[1] - pose_image_arr.shape[1]) // 2
                zero_end = zero_start + pose_image_arr.shape[1]
                canny_image_arr[:, zero_start:zero_end, :] = 0
        else:
            canny_image_resized = Image.plain_color(width, height, color=(0, 0, 0))

        return canny_image_resized, pose_image_resized







def recrop_control_images(self, width, height, canny_image=None, pose_image=None):

    
    if pose_image is not None:
        pose_image_resized = pose_image.resize(height=height)
        if pose_image_resized.size[0] < width:
            left_pad = (width - pose_image_resized.size[0]) // 2
            right_pad = width - pose_image_resized.size[0] - left_pad
            np_arr = np.full((height, width, 3), np.array([0,0,0]), dtype=np.uint8)
            np_arr[:, left_pad:width-right_pad, :] = pose_image_resized.np_arr
            pose_image_resized = Image.from_numpy(np_arr)
        elif pose_image_resized.size[0] > width:
            left_pad = (pose_image_resized.size[0] - width) // 2
            pose_image_resized = pose_image_resized.crop(left_pad, 0, width, height)
    else:
        pose_image_resized = Image.plain_color(width, height, color=(0, 0, 0))


    if canny_image is not None:
        canny_image_resized = canny_image.resize(height=height)
        if canny_image_resized.size[0] < width:
            left_pad = (width - canny_image_resized.size[0]) // 2
            right_pad = width - canny_image_resized.size[0] - left_pad
            np_arr = np.full((height, width, 3), np.array([0,0,0]), dtype=np.uint8)
            np_arr[:, left_pad:width-right_pad, :] = canny_image_resized.np_arr
            canny_image_resized = Image.from_numpy(np_arr)
        elif canny_image_resized.size[0] > width:
            left_pad = (canny_image_resized.size[0] - width) // 2
            canny_image_resized = canny_image_resized.crop(left_pad, 0, width, height)
        
        if pose_image is not None:
            canny_image_arr = canny_image_resized.np_arr
            pose_image_arr = pose_image_resized.np_arr

            zero_start = (canny_image_arr.shape[1] - pose_image_arr.shape[1]) // 2
            zero_end = zero_start + pose_image_arr.shape[1]
            canny_image_arr[:, zero_start:zero_end, :] = 0
    else:
        canny_image_resized = Image.plain_color(width, height, color=(0, 0, 0))

    return canny_image_resized, pose_image_resized

        # if canny_image is not None:
        #     canny_image = canny_image.resize(height=height)

        #     if canny_image.size[0] < width:
        #         canny_image = canny_image.resize(width=width)
        #         size_diff = np.abs((height - canny_image.size[1])) // 2
        #         canny_image = canny_image.crop(0, size_diff, width, height)
        #     elif canny_image.size[0] > width:
        #         size_diff = np.abs(canny_image.size[0] - width) // 2
        #         canny_image = canny_image.crop(size_diff, 0, width, height)
        # else:
        #     canny_image = Image.plain_color(width, height, color=(0, 0, 0))
        
        # if pose_image is not None:
        #     pose_image = pose_image.resize(height=height)
        #     if pose_image.size[0] > width:
        #         size_diff = np.abs(pose_image.size[0] - width) // 2
        #         pose_image = pose_image.crop(size_diff, 0, width, height)

        # # pose_image = pose_image.resize(height=height)

        #     pose_image_arr = pose_image.np_arr.copy()
        #     canny_image_arr = canny_image.np_arr.copy()


        #     # zero out middle columns of image where pose will be overlayed
        #     zero_start = (canny_image_arr.shape[1] - pose_image_arr.shape[1]) // 2
        #     zero_end = zero_start + pose_image_arr.shape[1]
        #     canny_image_arr[:, zero_start:zero_end, :] = 0

        #     # canny_image = canny_image[:, :, None]
        #     # canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
        #     croped_canny_image = Image.from_numpy(canny_image_arr)
            

        #     dialated_pose_arr = np.zeros_like(canny_image_arr)
        #     dialated_pose_arr[:, zero_start:zero_end, :] = pose_image_arr
        #     dialated_pose_image = Image.from_numpy(dialated_pose_arr)
        # else:
        #     croped_canny_image = canny_image
        #     dialated_pose_image = Image.plain_color(width, height, (0, 0, 0))

        # return croped_canny_image, dialated_pose_image