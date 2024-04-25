from abc import ABC, abstractmethod



class MediaDownloadException(Exception):
    pass

class BaseAdapter(ABC):


    @abstractmethod
    def get_metadata(self, url):
        pass
    
    @abstractmethod
    def download_file(self, url, destination):
        pass