import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import shutil


from config import MOCK_BACKEND_FOLDER



# def pack_args(**args):
    # return {k: v for k, v in args.items() if v is not None}

class BackendMock:

    def __init__(self, namespace, folder=MOCK_BACKEND_FOLDER):
        folder_path = Path(folder)
        if not folder_path.exists():
            raise Exception("base folder is not existing")
        namespace_path = folder_path / namespace
        if not namespace_path.exists():
            os.mkdir(namespace_path)
        if not (namespace_path / 'backend').exists():
            os.mkdir(namespace_path / 'backend')
        self.folder = folder
        self.namespace = namespace
        self.dataframes = {}
        self._load_dataframes()
        self.namespace_path = namespace_path


    def _get_file_path(self, data_type):
        return os.path.join(self.folder, self.namespace, 'backend', f"{data_type}.csv")
    


    def _load_dataframes(self):
        for data_type in ['movies', 'media', 'news_events']:
            file_path = self._get_file_path(data_type)
            if os.path.exists(file_path):
                self.dataframes[data_type] = pd.read_csv(file_path)
            else:
                self.dataframes[data_type] = pd.DataFrame()


    def _save_dataframe(self, data_type):
        self.dataframes[data_type].to_csv(self._get_file_path(data_type), index=False)

    def _delete_dataframe(self, data_type):
        if (os.path.exists(self._get_file_path(data_type))):
            os.remove(self._get_file_path(data_type))

    # ... other methods like create_movie, get_movie ...

    def add_media(
            self,
            name: str, 
            type: str, 
            source: str, 
            video_duration: float = None, 
            width: int = None, 
            height: int = None, 
            is_processed: bool = None,
            account_id: str = None, 
            movie_context: str = None,
            url: str = None,
            thumbnail: str = None,
            faces=None, 
            events=None,
            media_id=None,
        ):
        # Create a new media entry    
        new_media = {
            "id": media_id,
            "name": name,
            "type": type,
            "source": source,
            "height": height,
            "width": width,
            "isProcessed": is_processed,
            "videoDuration": video_duration,
            "movieContext": movie_context,
            "url": url,
            "thumbnail": thumbnail,
            "faces": faces,
            "events": events,            
        }
        # Add additional fields from kwargs
        self.dataframes['media'] = self.dataframes['media'].append(new_media, ignore_index=True)
        self._save_dataframe('media')
        return new_media

    def get_media(self, media_id: str):
        # Retrieve a media entry
        media = self.dataframes['media']
        media_entry = media[media['id'] == media_id]
        if not media_entry.empty:
            return media_entry.to_dict(orient='records')[0]
        else:
            return None

    def delete_media(self, media_id: str):
        # Delete a media entry
        media = self.dataframes['media']
        self.dataframes['media'] = media[media['id'] != media_id]
        self._save_dataframe('media')
        return {"status": "deleted", "id": media_id}
    

    def delete_all_media(self, media_id: str):
        # Delete all media entries
        records = self.dataframes['media'].to_dict(orient='records')
        self._delete_dataframe("media")
        self._load_dataframes()
        # self.dataframes['media'] = pd.DataFrame()
        # self._save_dataframe('media')
        return records
        # return {"status": "deleted", "id": media_id}
        
    

    def delete_namespace(self):
        if os.path.exists(self.namespace_path):
            shutil.rmtree(self.namespace_path)
            print(f"namespace '{self.namespace}' and all its contents have been deleted.")
        else:
            print(f"Folder '{self.namespace_path}' does not exist.")




    def create_news_event(
            self,
            id,
            data, 
            text, 
            facts, 
            opinions, 
            date, 
            platform, 
            channel, 
            platform_id, 
            type, 
            tags, 
            reliability, 
            url, 
            media_ids,
            metrics
        ):
        pass


    def get_news_event_by_id(self, id):
        pass



    def get_news_events(self, ids=None, offset=0, limit=10, from_date=None, to_date=None, search=None):
        pass


    def update_news_event(
            self,
            id, 
            data=None, 
            text=None, 
            facts=None, 
            opinions=None, 
            date=None, 
            platform=None, 
            channel=None,
            platform_id=None, 
            type=None, 
            tags=None, 
            reliability=None, 
            url=None, 
            media_id=None,
            metrics=None
        ):
        pass