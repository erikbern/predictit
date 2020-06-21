import json
import requests

class PredictItScraper:
    def __init__(self):
        self.session = requests.Session()
        self.access_token = None

    def login(self):
        credentials = json.load(open('credentials.json'))
        res = self.session.post('https://www.predictit.org/api/Account/token',
                                data={'grant_type': 'password', 'rememberMe': 'true', **credentials})
        res.raise_for_status()
        self.access_token = res.json()['access_token']

    def get(self, path):
        if self.access_token:
            headers = {'Authorization': 'Bearer %s' % self.access_token}
        else:
            headers = {}
        res = self.session.get('https://www.predictit.org' + path, headers=headers)
        res.raise_for_status()
        return res.json()

