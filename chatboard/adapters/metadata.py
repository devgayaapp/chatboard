


class MetaData():

    def __init__(self):
        self.mimetype = None
        self.extension = None
        self.title = None
        self.created_time = None
        self.size = None
        self.is_complete = False


    def parse_google_metadata(self, m):
        self.mimetype = m['mimeType']
        self.extension = m['fileExtension']
        self.title = m['title']
        self.created_time = m['createdDate']
        self.size = m['fileSize']
        self.is_complete = True


    def parse_url_metadata(self, url: str):        
        if 'jpg' in url:
            self.mimetype = 'image/jpeg'
        self.is_complete = True

    def parse_response_metadata(self, response):
        self.mimetype = response.headers['Content-Type']
        self.is_complete = True