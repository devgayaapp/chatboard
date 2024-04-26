from urllib.parse import urlparse
import pandas as pd
from glob import glob
from bs4 import BeautifulSoup
from components.scrapping.labeling_utils import print_article
from config import AWS_SCRAPING_BUCKET, AWS_SCRAPING_BRONZE_BUCKET, AWS_SCRAPING_SILVER_BUCKET, DATA_DIR, DEBUG
from util.boto import get_s3_obj, list_s3_keys, upload_s3_file


from botocore.exceptions import ClientError
import io

if DEBUG:
    import ipywidgets as widgets
    from ipywidgets import Button, Layout, interactive

def url2filename(url):
    name = '-'.join((urlparse(url).hostname + urlparse(url).path).split('/'))
    return name

def get_scraping_raw_keys(location='s3', not_processed=False):
    excluded_keys = set()
    if not_processed:
        excluded_keys = set([k[:-4] for k in get_scraping_bronze_keys(location)])        
    if location == 's3':
        raw_data_keys = list_s3_keys(AWS_SCRAPING_BUCKET)
        raw_data_keys = [k for k in raw_data_keys if k[:-4] not in excluded_keys]
        return raw_data_keys
    elif location == 'local':
        raw_data_keys = glob(str(DATA_DIR / 'scrapping' / 'raw' / '*.txt'))
        raw_data_keys = [k for k in raw_data_keys if k not in excluded_keys]
        return raw_data_keys
        

def get_scraping_bronze_keys(location='s3', not_processed=False):
    excluded_keys = set()
    if not_processed:
        excluded_keys = set([k[:-4] for k in get_scraping_silver_keys(location)])
    if location == 's3':
        bronze_data_keys = list_s3_keys(AWS_SCRAPING_BRONZE_BUCKET)
        bronze_data_keys = [k for k in bronze_data_keys if k not in excluded_keys]
        return bronze_data_keys
    elif location == 'local':
        raw_data_keys = glob(str(DATA_DIR / 'scrapping' / 'unlabeled' / '*.csv'))
        raw_data_keys = [k[:-4] for k in raw_data_keys if k not in excluded_keys]
        return raw_data_keys


def get_scraping_silver_keys(location='s3'):
    if location == 's3':
        raw_data_keys = list_s3_keys(AWS_SCRAPING_SILVER_BUCKET)
        return raw_data_keys
    elif location == 'local':
        silver_data_keys = list_s3_keys(AWS_SCRAPING_SILVER_BUCKET)
        return silver_data_keys


def upload_bronze_df(df, url, upload=True):
    name = url2filename(url)
    filepath = DATA_DIR / 'scrapping' / 'unlabeled' / f'{name}.csv' 
    df.to_csv(filepath, index=False)
    if upload:
        upload_s3_file(filepath, AWS_SCRAPING_BRONZE_BUCKET)


def upload_silver_df(df, url):
    name = url2filename(url)
    filepath = DATA_DIR / 'scrapping' / 'labeled' / f'{name}.csv' 
    df.to_csv(filepath, index=False)
    upload_s3_file(filepath, AWS_SCRAPING_SILVER_BUCKET)
    

def get_scrapping_soup(key, location='s3'):
    if location == 's3':
        raw = get_s3_obj(AWS_SCRAPING_BUCKET, key)
        url = raw.readline().decode('utf-8').removesuffix('\n')
        soup = BeautifulSoup(raw, 'html.parser')
        return url, soup
    elif location == 'local':
        with open(key, 'r') as f:
            url = f.readline().removesuffix('\n')
            soup = BeautifulSoup(f, 'html.parser')
            return url, soup


def get_scrapping_labeled_data(location='s3'):
    if location == 's3':
        df = pd.DataFrame()
        for key in get_scraping_silver_keys(location):
            buf = get_s3_obj(AWS_SCRAPING_SILVER_BUCKET, key)
            df = pd.concat([df, pd.read_csv(buf)])
        return df
    elif location == 'local':
        df = pd.DataFrame()
        for f in glob(str(DATA_DIR / 'scrapping' / 'labeled' / '*.csv')):
            df = pd.concat([df, pd.read_csv(f)])
        return df


def get_unlabeled_df(key, location='s3'):
    if location == 's3':
        buf = get_s3_obj(AWS_SCRAPING_BRONZE_BUCKET, key)
        df = pd.read_csv(buf)    
        return df
    elif location == 'local':
        # f = url2filename(key)
        return pd.read_csv(DATA_DIR / 'scrapping' / 'unlabeled' / f'{key}.csv')



def get_labeld_df(key, location='s3'):
    if location == 's3':
        buf = get_s3_obj(AWS_SCRAPING_SILVER_BUCKET, key)
        df = pd.read_csv(buf)    
        return df
    elif location == 'local':
        # f = url2filename(key)
        return pd.read_csv(DATA_DIR / 'scrapping' / 'labeled' / f'{key}.csv')


# def get_labeled_keys(location='s3'):
#     if location == 's3':
#         raw_data_keys = list_s3_keys(AWS_SCRAPING_SILVER_BUCKET)
#         return raw_data_keys

class SoupStoreException(Exception):
    pass


class SoupStore():

    def __init__(self, location='s3'):
        # self.location = location
        # self.raw_data_keys = get_scraping_raw_keys(location)
        # self.selected_key = self.raw_data_keys[0]      
        self.selected_key = -1

    def list(self, location='s3', not_processed=False):
        if not location in ['s3', 'local']:
            raise ValueError('location must be s3 or local')
    # def show(self):
        self.raw_data_keys = get_scraping_raw_keys(location, not_processed)
        self.selected_key = self.raw_data_keys[0]
        self.location = location

        def select(a):
            self.selected_key = a
            print(a)
            # global selected_key
            # self.selected_key = 0
            # selected_key = a
            # print(a)

        return interactive(select, a=widgets.Select(
            options=self.raw_data_keys,
            value=self.selected_key,
            # rows=10,
            description='keys:',
            disabled=False,
            layout=Layout(width='100%', height='180px')
        ))

    def get_selected_soup(self):
        url, soup = get_scrapping_soup(self.selected_key, self.location)
        return url, soup

    def selected_soup_stats(self):
        url, soup = get_scrapping_soup(self.selected_key, self.location)
        return {
            'h1': len(soup.find_all('h1')),
            'h2': len(soup.find_all('h2')),
            'p': len(soup.find_all('p')),
            'img': len(soup.find_all('img')),
        }





class BronzeFeatureStorage:


    def __init__(self, location='s3'):
        # self.location = location
        # self.raw_data_keys = get_scraping_raw_keys(location)
        # self.selected_key = self.raw_data_keys[0]      
        self.selected_key = -1
        self.location = location

    def list(self, location='s3', not_processed=False):
        if not location in ['s3', 'local']:
            raise ValueError('location must be s3 or local')
    # def show(self):
        self.raw_data_keys = get_scraping_bronze_keys(location, not_processed)
        self.selected_key = self.raw_data_keys[0]
        self.location = location

        def select(a):
            self.selected_key = a
            print(a)
            # global selected_key
            # self.selected_key = 0
            # selected_key = a
            # print(a)

        return interactive(select, a=widgets.Select(
            options=self.raw_data_keys,
            value=self.selected_key,
            # rows=10,
            description='keys:',
            disabled=False,
            layout=Layout(width='100%', height='180px')
        ))

    def get_df(self, selected_key=None, url=None):
        if url:
            selected_key = url2filename(url)+'.csv'
        if not selected_key:
            selected_key = self.selected_key
        return get_unlabeled_df(selected_key, self.location)






class SilverStorage:

    def __init__(self, location='s3'):
        # self.location = location
        # self.raw_data_keys = get_scraping_raw_keys(location)
        # self.selected_key = self.raw_data_keys[0]      
        self.selected_key = -1
        self.location = location

    def list(self, location='s3', not_processed=False):
        if not location in ['s3', 'local']:
            raise ValueError('location must be s3 or local')
    # def show(self):
        # self.raw_data_keys = get_labeled_keys(location)
        self.raw_data_keys = get_scraping_silver_keys(location)
        self.selected_key = self.raw_data_keys[0]
        self.location = location

        def select(a):
            self.selected_key = a
            print(a)
            # global selected_key
            # self.selected_key = 0
            # selected_key = a
            # print(a)

        return interactive(select, a=widgets.Select(
            options=self.raw_data_keys,
            value=self.selected_key,
            # rows=10,
            description='keys:',
            disabled=False,
            layout=Layout(width='100%', height='180px')
        ))


    def get_df(self, selected_key=None, url=None):
        if url:
            selected_key = url2filename(url)+'.csv'
        if not selected_key:
            selected_key = self.selected_key
        return get_labeld_df(selected_key, self.location)


    def present_article(self, selected_key=None, url=None, img=True, use_prediction=False):        
        df = self.get_df(selected_key=selected_key, url=url)
        return print_article(df, img=img, use_prediction=use_prediction)


    def add(self, df, url):
        upload_silver_df(df, url)

    def get_training_df(self):
        return get_scrapping_labeled_data(self.location)

        