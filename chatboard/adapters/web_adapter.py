import cv2
import requests
import numpy as np
from urllib.parse import urlparse
from components.adapters.base_adapter import BaseAdapter, MediaDownloadException
from components.adapters.metadata import MetaData
from config import VIDEO_DIR, IMG_DIR
import regex
import uuid


class NotImplementedMime(Exception):
    pass


class TitleNotFoundException(Exception):
    pass


def check_if_contains_extention(url):
    if 'jpg' in url:
        return 'jpg'
    elif 'png' in url:
        return 'png'
    elif 'jpeg' in url:
        return 'jpeg'
    elif 'webp' in url:
        return 'webp'
    return None

def generate_random_filename(url):
    return str(uuid.uuid4()) + '.' + check_if_contains_extention(url) 

FILENAME = '[\w-]+\.(jpg|png|txt|jpeg)'
# mimetypes.types_map.keys()
class WebAdapter(BaseAdapter):

    def __init__(self):
        self.filename_re = regex.compile(FILENAME)
        pass
    
    def _get_file_name_from_url(self, url, mimetype=None):
        pathname = urlparse(url).path.replace('%','')
        if res := self.filename_re.search(pathname):
            return res.group(0)
        if not url.endswith(('jpg','jpeg','png','webm')):
            ext = check_if_contains_extention(url)
            if ext:
                return str(uuid.uuid4()) + '.' + ext 
        sanitized_name = pathname[1:]
        if mimetype:
            if mimetype == 'image/jpeg':
                return sanitized_name + '.jpg'
        return sanitized_name
    
    def get_metadata(self, url):          
        metadata = {}
        
        if '.jpg' in url or '.png' in url or '.jpeg' in url or '.webp' in url:
            metadata['mimeType'] = 'image/jpeg'
            metadata['title'] = self._get_file_name_from_url(url)
        # elif 'image' in url or 'photo' in url:
        #     metadata['mimeType'] = 'image/jpeg'
        #     metadata['title'] = self._get_file_name_from_url(url)
        # elif ('.png') in url:
        #     raise NotImplementedMime('png not implemented')

        elif ('video') in url:
            raise NotImplementedMime('video not implemented')
        else:
            raise NotImplementedMime('unknown url type. need to implement')
        
        return metadata

    
    def download_file(self, url, destination: str):         
        # image_url  = urlparse(image_path)
        metadata = self.get_metadata(url)
        if metadata['mimeType'] == 'image/jpeg':
            return self.download_image(url, destination, metadata)


    def download_image(self, url, destination: str, metadata: MetaData):
        destination = IMG_DIR if not destination else destination
        headers = {'User-Agent': 'Musbot/0.1'}
        res = requests.get(url, headers=headers)
        if res.status_code != 200:
            raise MediaDownloadException(f'media download status: {res.status_code} for link {url}') 
        arr = np.asarray(bytearray(res.content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        if len(img.shape) == 2:
            height, width = img.shape
            layers = 1
        else:
            height, width, layers = img.shape
        metadata = {
            'height': height, 
            'width': width, 
            'layers': layers,
            'mimeType': res.headers['content-type'],
            'title': self._get_file_name_from_url(url, res.headers['content-type']),
            }
        filename = destination / metadata['title']
        cv2.imwrite(str(filename), img)
        return filename, metadata