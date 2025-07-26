import os
from typing import Any, Dict, Optional
import requests
import json
import uuid

from src.models.model_api import APIProvider


class HiveAPIProvider(APIProvider):
    """
        Starting page: https://docs.thehive.ai/docs/getting-started
        API doc: https://docs.thehive.ai/reference/ai-generated-image-and-video-detection-1
        API usage example: https://docs.thehive.ai/reference/submit-a-task-synchronously 

        Source classification:
            sora, pika, haiper, kling, luma, hedra, runway, hailuo, mochi, flux, hallo, hunyuan, recraft, leonardo, luminagpt, 
            var, liveportrait, mcnet, pyramidflows, sadtalker, aniportrait, cogvideos, makeittalk, sdxlinpaint, stablediffusioninpaint, 
            bingimagecreator, adobefirefly, lcm, dalle, pixart, glide, stablediffusion, imagen, amused, stablecascade, midjourney, 
            deepfloyd, gan, stablediffusionxl, vqdiffusion, kandinsky, wuerstchen, titan, ideogram, sana, emu3, omnigen, flashvideo, 
            transpixar, cosmos, janus, dmd2, switti, 4o, grok, wan, infinity, veo3, imagen4, other_image_generators (image generator other than those that have been listed), 
            inconclusive, inconclusive_video (no video source identified), or none (media is not AI-generated)
    """
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("HIVE_API_KEY")
        self.endpoint = 'https://api.thehive.ai/api/v3/task/sync' #"https://api.hivemoderation.com/api/v2/task/sync"
        # Initialize any other required attributes here

    def predict(self, image_path: str, user_id: str = None, post_id: str = None) -> Dict[str, Any]:
        if user_id is None:
            user_id = str(uuid.uuid4())
        if post_id is None:
            post_id = str(uuid.uuid4())

        headers = {
            'Authorization': f'Token {self.api_key}',
            # 'Content-Type': 'application/json'
        }
        # For local files, we need to upload the file. The API supports file uploads via multipart/form-data.
        # But the docs show url-based submission. We'll use file upload for local images.
        # https://docs.thehive.ai/reference/submit-a-task-synchronously
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'user_id': user_id,
                'post_id': post_id,
            }
            response = requests.post(self.endpoint, headers=headers, files=files, data=data)
        try:
            response_str = response.text
            return response_str.json()
        except Exception:
            return {"error": "Invalid response", "status_code": response.status_code, "text": response.text}


class SightengineAPIProvider(APIProvider):
    """
    API doc: https://sightengine.com/docs/ai-generated-image-detection
    API usage example: https://sightengine.com/docs/api-reference/ai-generated-image-detection
    """
    def __init__(self):
        super().__init__()
        self.api_user = os.environ.get("SIGHTENGINE_API_USER")
        self.api_secret = os.environ.get("SIGHTENGINE_API_SECRET")
        self.endpoint = "https://api.sightengine.com/1.0/check.json"

    @staticmethod
    def extract_score(self, response_json) -> float:
        return response_json.get("type")["ai_generated"]


    def predict(self, image_path: str, 
                *, 
                score_only: bool = False
                ) -> Dict[str, Any]:
        """
        Raw image only
        """
        files = {'media': open(image_path, 'rb')}
        data = {
            'models': 'genai',
            'api_user': self.api_user,
            'api_secret': self.api_secret
        }
        response = requests.post(self.endpoint, files=files, data=data)
        result = response.json()
        try:
            if score_only:
                return self.extract_score(result)
            return result
        except Exception:
            return {"error": "Invalid response", "status_code": response.status_code, "text": response.text}
        


class OpenAI(APIProvider):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.provider = "openai"
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_base = os.environ.get("OPENAI_API_BASE")
        self.api_version = os.environ.get("OPENAI_API_VERSION")
        self.api_type = os.environ.get("OPENAI_API_TYPE")