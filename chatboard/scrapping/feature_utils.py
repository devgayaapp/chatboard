from glob import glob
import io
from operator import index
import pandas as pd
from components.scrapping.html_parser import tag_component
from config import DATA_DIR, AWS_SCRAPING_SILVER_BUCKET, DEBUG
from util.boto import get_s3_obj, list_s3_keys, upload_s3_file
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from config import DATA_DIR
import re
import numpy as np


RELEVANT_TAGS = ['h1', 'h2', 'p', 'img', 'div', 'span', 'h3' , 'h4']

TAGS = ['']


def get_attr_set(elements):
    attr_set = set([])
    for e in elements:
        attr_set = attr_set.union(e.attrs.keys())
        # attr_set.add(e.attrs.keys())
    return attr_set

def get_hpi_elements(soup):
    elements = []
    for p in soup.find_all(RELEVANT_TAGS):
        elements.append(p)
    return elements



def get_all_soups():
    soups = []
    for f in glob(str(DATA_DIR / 'scrapping' / 'raw' / '*.txt')):
        with open(f, 'r') as file:
            url = file.readline().removesuffix('\n')
            soup = BeautifulSoup(file.read(), 'html.parser')
            soups.append({
                'soup': soup,
                'url': url,
                'file': f,
            })
    return soups



def sibiling_data(p):
    ns = p.find_next_sibling()
    ps = p.find_previous_sibling()
    return {
        'next_sibling_tag': ns.name if ns else None,
        'next_sibling_text': ns.text if ns else None,
        'previous_sibling_tag': ps.name if ps else None,
        'previous_sibling_text': ps.text if ps else None,
    }

def get_g_parent_attr(p, idx ,tag):
    for i in range(idx):
        if p.parent is None:
            return None
        p = p.parent
    return p.attrs.get(tag, None)

def get_g_parent_name(p, idx):
    for i in range(idx):
        if p.parent is None:
            return None
        p = p.parent
    return p.name

def parent_children_data(p):
    parent = p.find_parent()
    parent_children = parent.findChildren()
    self_children = p.findChildren()
    len([c for c in parent_children if c.name == 'h1'])
    return {
        'depth': len(p.findParents()),
        'children_count': len(self_children),
        'parent_tag': parent.name if parent else None,
        'parent_text': parent.text if parent else None,
        'parent_children_count': len(p.findChildren()),
        'parent_h1_number': len([c for c in parent_children if c.name == 'h1']),
        'g0_parent_class': str(parent.attrs.get('class', None)),
        'g0_parent_id': parent.attrs.get('id', None),
        'g0_parent_alt': parent.attrs.get('alt', None),
        'g0_parent_tag': parent.name if parent else None,
        'g1_parent_class': str(get_g_parent_attr(p, 2, 'class')),
        'g1_parent_id': get_g_parent_attr(p, 2, 'id'),
        'g1_parent_alt': get_g_parent_attr(p, 2, 'alt'),
        'g1_parent_tag': get_g_parent_name(p, 2),
        'g2_parent_class': str(get_g_parent_attr(p, 3, 'class')),
        'g2_parent_id': get_g_parent_attr(p, 3, 'id'),
        'g2_parent_alt': get_g_parent_attr(p, 3, 'alt'),
        'g2_parent_tag': get_g_parent_name(p, 3),
        'g3_parent_class': str(get_g_parent_attr(p, 4, 'class')),
        'g3_parent_id': get_g_parent_attr(p, 4, 'id'),
        'g3_parent_alt': get_g_parent_attr(p, 4, 'alt'),
        'g3_parent_tag': get_g_parent_name(p, 4),
        'parent_h2_count': len([c for c in parent_children if c.name == 'h2']),
        'parent_h3_count': len([c for c in parent_children if c.name == 'h3']),
        'parent_h3_count': len([c for c in parent_children if c.name == 'h4']),
        'parent_p_count': len([c for c in parent_children if c.name == 'p']),
        'parent_img_count': len([c for c in parent_children if c.name == 'img']),
        'children_h1_count': len([c for c in self_children if c.name == 'h1']),
        'children_h2_count': len([c for c in self_children if c.name == 'h2']),
        'children_h3_count': len([c for c in self_children if c.name == 'h3']),
        'children_h3_count': len([c for c in self_children if c.name == 'h4']),
        'children_p_count': len([c for c in self_children if c.name == 'p']),
        'children_img_count': len([c for c in self_children if c.name == 'img']),
    }


def get_number(value):
    try:
        if value.endswith('px'):
            return int(value.removesuffix('px'))
        return float(value)
    except:
        return None

def get_attr_count(p, attrname):
    return len(p.attrs.get(attrname, []))

def get_attr_len(p, attrname):
    return len(p.attrs.get(attrname, 0))
    # c = p.attrs.get(attrname, [None])
    # if c is None:
    #     return 0
    # return len(c)

def sanitize_url(img_url, url, logger=None):
    p_img_url = urlparse(img_url)
    p_url = urlparse(url)
    if not p_img_url.scheme in  ['http', 'https']:        
        if img_url.startswith('//'):
            return 'https:' + img_url
        if img_url.startswith('/'):
            return p_url.scheme + '://' + p_url.hostname + img_url
        else:
            if logger:
                logger.warn(f'Invalid url {img_url}')
    return img_url

def get_feature_df(soup, url, logger=None):
    entries = []
    for p in soup.find_all(RELEVANT_TAGS):
        try:
            img_url = p.attrs.get('src', None)
            if img_url:
                img_url = sanitize_url(img_url, url, logger)
            tag = tag_component(p)            
            features = {
                'prediction': tag,
                'label': 'not labeled',
                'elements': str(p),
                'tag': p.name,
                'text': p.text,
                'text_len': len(p.text),
                'source': url,
                'class_count': len(p.attrs.get('class', [])),
                'alt_len': len(p.attrs.get('alt', '')),
                'alt': p.attrs.get('alt', None),
                'class': str(p.attrs.get( 'class', None)),
                'data-bind': p.attrs.get('data-bind', None),
                'data-index': p.attrs.get('data-index', None),
                'data-sub-count': p.attrs.get('data-sub-count', None),
                'draggable': p.attrs.get('draggable', None),
                'height': get_number(p.attrs.get('height', None)),
                'id': p.attrs.get('id', None),
                'itemprop': p.attrs.get('itemprop', None),
                'loading': p.attrs.get('loading', None),
                'name': p.attrs.get('name', None),
                'nopin': p.attrs.get('nopin', None),
                'role': p.attrs.get('role', None),
                'src': img_url,
                'style': p.attrs.get('style', None),
                'title': p.attrs.get('title', None),
                'v-show': p.attrs.get('v-show', None),
                'width': get_number(p.attrs.get( 'width', None)),
            }
            features.update(sibiling_data(p))
            features.update(parent_children_data(p))
            entries.append(features)
        except Exception as e:
            if logger:
                logger.error(f'Error processing {url} {e}')
            raise e
    df = pd.DataFrame(entries)
    return df

def save_df(df, url):
    name = '-'.join((urlparse(url).hostname + urlparse(url).path).split('/'))
    df.to_csv(DATA_DIR / 'scrapping' / 'unlabeled' / f'{name}.csv', index=False)
    # save_df(df, url)



def get_prediction_features_df(df, clf_objs):
    min_depth = clf_objs['min_depth']
    max_depth = clf_objs['max_depth']
    
    feature_df = df[['tag', 'text','label']].fillna('')
    feature_df['class_count'] = df['class_count'].fillna(0)
    feature_df['depth'] = df['depth'].fillna(0)
    feature_df['children_count'] = df['children_count'].fillna(0)
    feature_df['height'] = df['height'].fillna(-1)
    feature_df['width'] = df['width'].fillna(-1)
    feature_df['length'] = feature_df['text'].apply(lambda x: len(x))
    feature_df['relative_depth'] = feature_df['depth'].apply(lambda x: (x - min_depth) / (max_depth - min_depth))
    feature_df['src'] = df['src'].fillna('')
    feature_df['new_line_count'] = feature_df['text'].apply(lambda x: len(re.findall('\n', x)))
    # feature_df['sub_div_count'] = df['text'].apply(lambda x: len(re.findall('<div', x)))


    return feature_df

def get_first_article(df, logger=None):
    article_df = df[df['prediction'] == 'article_title']
    if len(article_df) > 1:
        if logger:
            logger.info(f'Found {len(df)} articles. taking first one.')
        return df[0: article_df.index[1]]
    else: 
        return df


def filter_small(df, logger=None):
    if logger:
        logger.warn(f'Filtering small elements')
    return df[df['length'] > 10]

def filter_paragraphs(df, logger=None):
    pdf = df[(df['prediction'] == 'paragraph') & (df['length'] < 20)]
    if len(pdf) > 0:        
        if logger:
            logger.warn(f'Found {len(pdf)} paragraphs with less than 20 characters. Filtering them.')
        return df.drop(pdf.index).reset_index(drop=True)
    return df

def filter_image_formats(df, logger=None):
    bad_formats_df = df[(df['prediction'] == 'image') & (~ df['src'].str.contains('|'.join(['jpg', 'jpeg', 'png', 'webp'])))]
    if len(bad_formats_df) > 0:
        if logger:
            logger.warn(f'Found {len(bad_formats_df)} svg images. removing them.')
        df = df.drop(bad_formats_df.index).reset_index(drop= True)
        return df
    return df

def filter_image_duplicates(df, logger=None):
    image_df = df[df['prediction'] == 'image']
    img_dict = {}
    bad_indexes = []
    for i, row in image_df.iterrows():
        if row['src'] in img_dict:
            bad_indexes.append(True)
        else:
            img_dict[row['src']] = True
            bad_indexes.append(False)
    if len(bad_indexes) > 0:
        if logger:
            logger.warn(f'Found {len(bad_indexes)} duplicate images. removing them.')
        bad_df = image_df[np.array(bad_indexes)]
        return df.drop(bad_df.index).reset_index(drop=True)
    return df


def verify_title(df, logger=None):
    title_prediction_count = len(df[df['prediction'] == 'title'])
    title_count = len(df[df['tag'] == 'h2'])
    h3_count = len(df[df['tag'] == 'h3'])
    tag = 'h2'
    if h3_count > title_count:
        tag = 'h3'
        title_count = h3_count
    if title_count > 0 and (title_prediction_count / title_count) < 0.5:
        if logger:
            logger.warn('No title found. Adding first paragraph as title.')
        df.loc[df['tag'] == tag, 'prediction'] = 'title'
    return df
    

def predict_transform_soup(soup, url, clf, logger=None, use_filter=True):
    clf_objs = clf.models[0].custom_objects

    df = get_feature_df(soup, url, logger)
    num2class = clf_objs['num2class']
    feature_df = get_prediction_features_df(df, clf_objs)

    if logger:
        logger.info(f'raw tags:')
        logger.info(df['tag'].value_counts())

    y_preds = clf.predict.run(feature_df)
    feature_df['prediction'] = y_preds
    feature_df['prediction'] = feature_df['prediction'].apply(lambda x: num2class[x])

    if logger:
        logger.info('all predictions:')
        logger.info(df['prediction'].value_counts())

    feature_df['tag'] = df['tag']

    if not use_filter:
        return feature_df


    filtered_df = get_first_article(feature_df, logger=logger)

    # filtered_df = verify_title(filtered_df, logger=logger)

    filtered_df = filtered_df[filtered_df['prediction'] != 'unknown'].reset_index(drop= True)

    # has_more_then_800_chars = sum(filtered_df['length'] > 800) > 0
    # if has_more_then_800_chars:
    #     if logger:
    #         logger.warn(f'Found {sum(filtered_df["length"] > 800)} elements with more then 800 chars.')
    #     filtered_df = filtered_df[filtered_df['length'] < 800].reset_index(drop= True)
    
    filtered_df = filter_paragraphs(filtered_df, logger=logger)
    #remove unknown media types
    filtered_df = filter_image_formats(filtered_df, logger=logger)
    filtered_df = filter_image_duplicates(filtered_df, logger=logger)
    
    if logger:
        logger.info('filtered predictions:')
        logger.info(filtered_df['prediction'].value_counts())

    return filtered_df




    # elif location == 'local':
    #     raw_data_keys = glob(str(DATA_DIR / 'scrapping' / 'raw' / '*.txt'))
    #     return raw_data_keys


def get_saved_feature_df(key, location='s3'):
    if location == 's3':
        buf = get_s3_obj(AWS_SCRAPING_SILVER_BUCKET, key)
        df = pd.read_csv(buf)    
        return df
