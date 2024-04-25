from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse
from components.paragraph import Paragraph, WebArticle, Media, RequiredParamMissingException, Title
from components.adapters.web_adapter import NotImplementedMime
import numpy as np

IMG_MAX_HEIGHT = 100000
IMG_MIN_HEIGHT = 300

IMG_MAX_WIDTH = 100000
IMG_MIN_WIDTH = 400



def validate_image(src):
        if not src:
            return False
        if 'jpg' in src or 'jpeg' in src:
            return True
        return False


def classify_is_image(p, min_height=IMG_MIN_HEIGHT, max_height=IMG_MAX_HEIGHT, min_width=IMG_MIN_WIDTH, max_width=IMG_MAX_WIDTH):
    height = p.get('height', None)
    width = p.get('width', None)
    if validate_image(p.get('src', None)) and height and width:
        if min_height <= int(height) <= max_height and min_width <= int(width) <= max_width:
            return 1
    return 0




def tag_component(p):    
    if p.name == 'img' and classify_is_image(p):        
        return 'image'
        
    if p.name in ('h1', 'h2', 'h3'):
        return 'title'
    if p.name == 'p' and p.text != '' and len(p.text) > 200:
        return 'paragraph'
    
    return 'unknown'
