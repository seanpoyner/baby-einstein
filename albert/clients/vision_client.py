# vision_client.py

import requests
import urllib3

class VisionClient:
    def __init__(self, url: str = "http://localhost:8000/sight/"):
        self.url = url
        # Disable insecure request warnings (optional).
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def get_caption(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(self.url, files=files, verify=False)
        if response.ok:
            print("Server Response:", response.status_code, response.reason)
            return response.json()
        else:
            print("Error:", response.status_code, response.text)
            return {}
