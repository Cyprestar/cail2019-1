import requests

url = 'http://127.0.0.1:5000/text_rank'


def text_rank(text):
    r = requests.post(url, data=text.encode('utf-8'))
    return r.text
