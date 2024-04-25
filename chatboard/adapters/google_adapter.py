from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from urllib import parse
from config import VIDEO_DIR
from components.adapters.metadata import MetaData
from components.adapters.base_adapter import BaseAdapter


class GoogleAdapter(BaseAdapter):

    def __init__(self):
        self.gauth = GoogleAuth()
        # self.gauth.LoadCredentialsFile('google_creds.txt')
        self.gauth.LoadCredentialsFile('google_creds.txt')
        # self.gauth.LocalWebserverAuth()
        self.drive = GoogleDrive(self.gauth)


    def get_id_from_url(self, url):
        return parse.urlparse(url).path.split('/')[3]


    def get_metadata(self, url):
        if type(url) == str:
            url = parse.urlparse(url)
        file_id = url.path.split('/')[3]
        assert len(file_id) == 33
        f = self.drive.CreateFile({'id': file_id})
        f.FetchMetadata()    
        return self._copy_metadata(f.metadata)

    def _copy_metadata(self, metadata):
        m = {}
        m.update(metadata)
        m['height'] = metadata['videoMediaMetadata']['height']
        m['width'] = metadata['videoMediaMetadata']['width']
        m['duration'] = int(metadata['videoMediaMetadata']['durationMillis']) / 1000
        del m['videoMediaMetadata']
        return m
    
    def get_id_from_url(self, url):
        return url.split('/')[-1]

    def download_file(self, url, destination):
        dest = VIDEO_DIR if not destination else destination
        file_id = parse.urlparse(url).path.split('/')[3]
        f = self.drive.CreateFile({'id': file_id})
        f.FetchMetadata()
        filepath = dest / f.metadata['title']        
        if f.metadata['mimeType'] == 'video/mp4':
            f.GetContentFile(filepath)
        return filepath, self._copy_metadata(f.metadata)


    def get_file_ids_from_folder(self, folder_url):
        folder_id = self.get_id_from_url(folder_url)
        file_list = self.drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        return [f['id'] for f in file_list]
    

        