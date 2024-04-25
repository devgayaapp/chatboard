


from datetime import datetime
import json
import pickle
from config import DATA_DIR, PEXELS_API_KEY, TEMP_DIR
import requests
import urllib.parse
import pandas as pd 
from components.video.video import Video
from components.image.image import Image
from components.media.media import Media


pexels_video_url = "https://api.pexels.com/videos/search"
pexels_photo_url = "https://api.pexels.com/v1/search"


class PexelsException(Exception):
    pass


def build_pexels_video_url(
        search_term, 
        page=1,
        per_page=15, 
        orientation=None, # landscape, portrait or square,
        size=None, #large(4K), medium(Full HD) or small(HD).
    ):
    base_url = pexels_video_url
    params = {
        "query": search_term,
        "per_page": per_page,
        "page": page
    }
    if size:
        if size not in ("large", "medium", "small"):
            raise Exception("size should be one of large(4K), medium(Full HD) or small(HD).")
        params["orientation"] = "orientation"
    if orientation:
        if orientation not in ("landscape", "portrait", "square"):
            raise Exception("orienteation should be one of landscape, portrait or square")
        params["orientation"] = orientation

    query_string = urllib.parse.urlencode(params)
    final_url = base_url + "?" + query_string
    return final_url


# curl -H "Authorization: 563492ad6f91700001000001662ec1239554478794a63d2c3c551867" \
  
search_term = "cats"


def search_pexcel_media(search_term, media_type="video", orientation=None, page=1, per_page=15):
    res = requests.get(
        build_pexels_video_url(search_term, page=page, per_page=per_page, orientation=orientation),
        headers={
            "Authorization": PEXELS_API_KEY
        }
    )

    if res.status_code != 200:
        raise PexelsException(res.text)
    
    stats = {
        'status': 
        {
            "requests_remaining": res.headers["X-Ratelimit-Remaining"],
            "rate_limit": res.headers["X-Ratelimit-Limit"],
            "limit_rest": res.headers["X-Ratelimit-Reset"],
        } 
    }
    data = res.json()
    data.update(stats)
    return data


def select_video_picture(row):
    pictures_list = row['video_pictures']
    if isinstance(pictures_list, str):
        pictures_list = json.loads(pictures_list.replace("'", '"'))

    return pictures_list[0]['picture']



def get_media_df(df, screen_width = 1080, screen_height = 1920, to_dict=True):
    # df['thumbnail'] = df.apply(lambda x: x['video_pictures'][0]['picture'], axis=1)
    df['thumbnail'] = df.apply(select_video_picture, axis=1)
    exploded_video_df = df.explode('video_files')
    normalized_df = pd.json_normalize(exploded_video_df['video_files'])
    full_df = pd.concat([exploded_video_df.rename(columns={'id': 'main_id'}).drop(['video_files','width', 'height'], axis=1).reset_index(), normalized_df.reset_index()], axis=1)
    full_df['pixels'] = full_df['width'] * full_df['height']
    full_df['min_pixels_diff'] = (full_df['pixels'] - screen_width * screen_height).abs()
    min_idx = full_df.groupby('main_id')['min_pixels_diff'].idxmin()
    video_media_df = full_df.loc[min_idx]
    if to_dict:
        return video_media_df.to_dict(orient='records')
    return video_media_df


def download_pexel_item(media_rec):
    ext = media_rec['file_type'].split('/')[1]    
    filename = f"pexels_{media_rec['main_id']}.{ext}"
    thumbnail_filename = f"thumbnail_pexels_{media_rec['main_id']}.png"
    thumbnail = Image.from_url(media_rec['thumbnail'])
    if (TEMP_DIR / filename).exists():
        video = Video.from_file(TEMP_DIR / filename)
    else:
        video = Video.from_url(media_rec['link'], filename)
    media = Media(
        media=video,
        thumbnail=thumbnail,
        # caption=description,
        platform_id=media_rec['main_id'],
        # source='hasbara',
        source="pexels",
        # date=item['createdTime'],
        # size=item['size'],
        filename=filename,
        thumbnail_filename=thumbnail_filename,
        platform_type='pexels',
    )
    media._is_downloaded = True
    return media



pexels_history_path = DATA_DIR / 'pexels' / 'paxels_video.csv'
pexels_status_path = DATA_DIR / 'pexels' / 'paxels_video.pkl'

class PexelsManager:


    def __init__(self, filepath=None):
        self.filepath = filepath or pexels_history_path
        # self._history_df = pd.read_csv(filepath)
        self._history_df = None
        self._status = None

    def _load_status(self):
        with open(pexels_status_path, 'rb') as f:            
            self._status = pickle.load(f)

    def _save_status(self):
        with open(pexels_status_path, 'wb') as f:
            pickle.dump(self._status, f)

    @property
    def status(self):
        self._load_status()
        return self._status
    
    @status.setter
    def status(self, value):
        self._status = value
        self._save_status()

    @property
    def history(self):
        self.load()
        return self._history_df
    
    @history.setter
    def history(self, val):
        self._history_df = val
        self.save()

    def load(self):
        self._history_df = pd.read_csv(self.filepath)
        self._history_df['video_files'] = self._history_df['video_files'].apply(json.loads)
        self._history_df['video_pictures'] = self._history_df['video_pictures'].apply(json.loads)
        self._history_df['user'] = self._history_df['user'].apply(json.loads)

    def save(self):
        save_df = self._history_df.copy()
        save_df['video_files'] = save_df['video_files'].apply(json.dumps)
        save_df['video_pictures'] = save_df['video_pictures'].apply(json.dumps)
        save_df['user'] = save_df['user'].apply(json.dumps)
        save_df.to_csv(self.filepath, index=False)
        # self._history_df.to_csv(self.filepath, index=False)

    def append(self, raw_df):
        curr_date = datetime.now()
        dated_raw_df = raw_df.copy()
        dated_raw_df['date_created']  = curr_date
        try:            
            self.history = self.history.append(dated_raw_df, ignore_index=True)            
        except FileNotFoundError as e:
            self.history = dated_raw_df


    def clear_history(self):
        try:
            self.filepath.unlink()
        except FileNotFoundError as e:
            pass 


    def search(self, search_term: str, page=1, per_page=15, orientation=None):
        search_results = search_pexcel_media(
            search_term, 
            page=page, 
            per_page=per_page, 
            orientation=orientation,
        )
        self.status = search_results['status']
        raw_df = pd.DataFrame.from_records(search_results['videos'])
        raw_df['page'] = page
        raw_df['per_page'] = per_page
        raw_df['search_term'] = search_term
        self.append(raw_df)
        return raw_df
    

    def get_history_search_terms(self):
        search_terms = list(self.history['search_term'].unique())
        return search_terms
    

    def get_history(self, search_term=None):
        if search_term is None:
            return self.history
        history_raw_df = self.history
        raw_df = history_raw_df[history_raw_df['search_term'] == search_term]
        return raw_df