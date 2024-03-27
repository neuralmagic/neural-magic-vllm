import time

import requests


url = "http://localhost:5000/generate"

obj = {
    "prompt": "hello world", "max_tokens": 200
}

response = requests.post(url, json=obj)
print(response._content)