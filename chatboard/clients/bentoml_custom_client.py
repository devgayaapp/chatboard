import requests



class BentomlClient:


    def __init__(self, server_url) -> None:
        self.server_url = server_url


    def post(self, endpoint, data):
        return requests.post(f'{self.server_url}/{endpoint}', json=data)