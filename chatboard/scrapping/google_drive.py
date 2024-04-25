import hashlib
from io import BytesIO
import os
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import components.backend as backend
from googleapiclient.http import MediaIoBaseDownload
from components.image.image import Image
from components.media.media import Media
from components.text.translate import translate_text
from components.video.video import Video
from util.boto import upload_s3_obj
from config import AWS_IMAGE_BUCKET, AWS_VOICE_BUCKET, AWS_THUMBNAIL_BUCKET, TEMP_DIR
from pathlib import Path
import pickle
import os

# from src.config import settings
import json

from util.sanitizers import get_sanitized_filename

# Load credentials from JSON file
# FOLDER_ID = "14d0FX2jNOBmwiAuI_fCmKzu0kHmkhosu"

client_file = 'muspark-395909-263bea589c9b.json'

with open(client_file) as f:
    credentials = service_account.Credentials.from_service_account_info(
        json.load(f)
    )

# # Create a service instance
drive_service = build('drive', 'v3', credentials=credentials)
# Scope for Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

def service_initialization():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('muspark-395909-263bea589c9b.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service

# drive_service = service_initialization()

def move_video_to_s3(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    # fh = open(filename, 'wb')
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}%.")    
    print("file downloaded!")
    fh.seek(0)
    upload_s3_obj(AWS_IMAGE_BUCKET, filename, fh)
    print("file uploaded!")


def download_image_from_drive(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    # fh = BytesIO()
    filepath = TEMP_DIR / filename
    with open(filepath, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        fh.seek(0)
        return Image.from_file(str(filepath))
        # return Image.from_bytes(fh.getvalue())


def download_video_from_drive(file_id, filename):
    request = drive_service.files().get_media(fileId=file_id)
    # fh = BytesIO()
    filepath = TEMP_DIR / filename
    if os.path.exists(filepath):
        return filepath
        # return Video(filepath)
    with open(filepath, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
        fh.seek(0)
        return filepath
        # return Video(filepath)


def verifiy_image_extension(filename):
    if not filename.endswith('.png') and not filename.endswith('.jpg') and not filename.endswith('.jpeg'):
        filename = filename + '.png'
    return filename

def verifiy_video_extension(filename):
    if not filename.endswith('.mp4') and not filename.endswith('.mov'):
        filename = filename + '.mp4'
    return filename




def get_google_drive_item_data(folder_id, item_id):
    folder = drive_service.files().get(
        fileId=folder_id,
        fields="id, name, createdTime"
    ).execute()
    item = drive_service.files().get(
        fileId=item_id,
        fields="id, name, mimeType, thumbnailLink, size, imageMediaMetadata, videoMediaMetadata, createdTime"
    ).execute()
    item.update({
        'folder_id': folder_id,
        'folder_name': folder['name'],
        'platform_group': folder_id,
    })
    return item



def get_google_drive_items(media_list, folder_id, indent=0, pageSize=30, folder_name=None, filter_type=None, item_limit=None, max_size=None):
    # Call the Drive v3 API to list files in the current folder
    next_page_token = None
    while True:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=pageSize,
            fields="nextPageToken, files(id, name, mimeType, thumbnailLink, size, imageMediaMetadata, videoMediaMetadata, createdTime)",            
            pageToken=next_page_token,
            orderBy='createdTime',
        ).execute()
        next_page_token = results.get('nextPageToken', None)
        items = results.get('files', [])
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                item_folder_name = item['name'].split('/')[-1].strip()
                get_google_drive_items(media_list, item['id'], indent + 1, pageSize=pageSize, folder_name=item_folder_name, filter_type=filter_type, item_limit=item_limit)
            else:
                try:
                    item['size'] = int(item['size']) // 1000000
                    if max_size is not None and item['size'] > max_size:
                        continue
                except Exception as e:
                    print(e)
                    continue
                if filter_type == 'VIDEO' and 'video' not in item['mimeType']:
                    continue
                elif filter_type == 'IMAGE' and 'image' not in item['mimeType']:
                    continue
                if item_limit and len(media_list) >= item_limit:
                    return media_list  
                item.update({
                    'folder_name': folder_name,
                    'platform_group': folder_id,
                })              
                media_list.append(item)
        if not next_page_token or (item_limit and len(media_list) >= item_limit):
            break
    return media_list




def get_google_drive_flat_items(media_list, folder_id, indent=0, pageSize=30, folder_name=None, filter_type=None, item_limit=None, max_size=None, skip=0):
    # Call the Drive v3 API to list files in the current folder
    next_page_token = None
    while True:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents",
            pageSize=pageSize,
            fields="nextPageToken, files(id, name, mimeType, thumbnailLink, size, imageMediaMetadata, videoMediaMetadata, createdTime)",            
            pageToken=next_page_token,
            orderBy='createdTime',
        ).execute()
        next_page_token = results.get('nextPageToken', None)
        items = results.get('files', [])
        for item in items:
            if skip > 0:
                skip -= 1
                continue
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                continue    
            else:
                try:
                    item['size'] = int(item['size']) // 1000000
                    if max_size is not None and item['size'] > max_size:
                        continue
                except Exception as e:
                    print(e)
                    continue
                if filter_type == 'VIDEO' and 'video' not in item['mimeType']:
                    continue
                elif filter_type == 'IMAGE' and 'image' not in item['mimeType']:
                    continue
                if item_limit and len(media_list) >= item_limit:
                    return media_list  
                item.update({
                    'folder_name': folder_name,
                    'platform_group': folder_id,
                })              
                media_list.append(item)
        if not next_page_token or (item_limit and len(media_list) >= item_limit):
            break
    return media_list

                # thumbnail_name = file_name.split('.')[0] + '.' + "png"                
                # print(f"https://drive.google.com/uc?export=view&id={item['id']}", file_name, thumbnail_name)
                # thumbnail_image = Image.from_url(thumbnail)
                # if not only_images:                      
                    # yield None, thumbnail_image, file_name


def translate_to_english(text):
    try:
        text_lang = detect(text)
        translated=''
        if text_lang != 'en':            
            translated = translate_text('en', text)
            return translated['translatedText']
        return text
    except LangDetectException as e:
        return ''
    




def process_drive_items_metadata(items):
    is_list = True
    if type(items) != list:
        is_list = False
        items = [items]
    enriched_items = []
    for item in items:
        try:
            file_name = item['name'].split('.')[0]
            description = translate_to_english(file_name)
            topic = translate_to_english(item['folder_name']) if item['folder_name'] is not None else ''
            item_copy = item.copy()
            item_copy['topic'] = topic
            item_copy['description'] = description
            enriched_items.append(item_copy)
        except Exception as e:
            print(e)
    
    if not is_list:
        return enriched_items[0]
    return enriched_items
        
        # try:
        #     file_name_lang = detect(file_name)
        #     translated=''
        #     if file_name_lang != 'en':
        #         print('translating', file_name)
        #         translated = translate_text('en', file_name)
        #         item['description'] = translated['translatedText']
        #     print(file_name_lang, file_name, translated['translatedText'])
        # except LangDetectException as e:
        #     pass


def hash_file_name(file_name):
    content_hash = hashlib.md5(file_name.encode('utf-8')).hexdigest()
    return content_hash


def get_filename(item):    
    filename = get_sanitized_filename(item['description'] + hash_file_name(item['name']))
    ext = item['name'].split('.')[-1]
    return f"{filename}.{ext}"


def download_google_drive_item(item):
    filename = get_filename(item)
    description = f"{item['topic']}. {item['description']}"
    thumbnail_image = Image.from_url(item['thumbnailLink'])
    if 'video' in item['mimeType']:
        filepath = download_video_from_drive(item['id'], filename)
        video = Video(
            filepath=filepath,
            width=item['videoMediaMetadata']['width'],
            height=item['videoMediaMetadata']['height'],
            duration=int(item['videoMediaMetadata']['durationMillis']) / 1000,
        )
        media = Media(
            media=video,
            thumbnail=thumbnail_image,
            caption=description,
            platform_id=item['id'],
            # source='hasbara',
            source=item['folder_name'],
            date=item['createdTime'],
            size=item['size'],
            filename=filename,
            platform_type='google_drive',
            platform_group=item['platform_group']
        )
    elif 'image' in item['mimeType']:
        image = download_image_from_drive(item['id'], filename)
        media = Media(
            media=image,
            thumbnail=thumbnail_image,
            caption=description,
            platform_id=item['id'],
            # source=f'google_drive',
            source=item['folder_name'],
            date=item['createdTime'],
            size=item['size'],
            filename=filename,
            platform_type='google_drive',
            platform_group=item['platform_group']
        )
    media._is_downloaded = True
    return media
                



def get_google_drive_items_gen(media_list, media_type=None, start_index=None, end_index=None, only_images=False, only_videos=False, item_limit=None, folder_id=None):

    if media_type is not None and not media_type in ['VIDEO', 'IMAGE']:
        raise Exception("type must be VIDEO or IMAGE")    

    sub_media_list = media_list
    if end_index is not None:
        sub_media_list = sub_media_list[:end_index]
    if start_index is not None:
        sub_media_list = sub_media_list[start_index:]

    for media_item in sub_media_list:
        # if not media_type or media.get('type') == media_type:        
        if media_item.get('type') == 'IMAGE':
            image = download_image_from_drive(media_item['id'])
            thumbnail_image = Image.from_url(media_item['thumbnail'])
            yield image, thumbnail_image, media_item['file_name'],  media_item['folder_name'], 'IMAGE'
        elif media_item.get('type') == 'VIDEO':
            video = download_video_from_drive(media_item['id'], media_item['file_name'])
            thumbnail_image = Image.from_url(media_item['thumbnail'])    
            yield video, thumbnail_image, media_item['file_name'],  media_item['folder_name'], 'VIDEO'


# def get_google_drive_items_list(media, media_type=None):

#     if media_type is not None and not media_type in ['VIDEO', 'IMAGE']:
#         raise Exception("type must be VIDEO or IMAGE")
    
#         # if not media_type or media.get('type') == media_type:
#     if media.get('type') == 'IMAGE':
#         image = download_image_from_drive(media['id'])
#         thumbnail_image = Image.from_url(media['thumbnail'])
#         return image, thumbnail_image, media['file_name'], media['folder_name']      
#     elif media.get('type') == 'VIDEO':
#         pass



class GoogleDriveScraper:

    def __init__(self) -> None:
        self.media_list = []
        self._curr = 0

    def load(self, FOLDER_ID: str):
        self.media_list = get_google_drive_items([], FOLDER_ID)
    
    def __iter__(self):
        return self

    def __next__(self):
        if self._curr >= len(self.media_list):
            raise StopIteration
        media = download_image_from_drive(self.media_list[self._curr]['id'])
        self._curr += 1
        return media

    def __getitem__(self, index):
        return download_image_from_drive(self.media_list[index]['id'])