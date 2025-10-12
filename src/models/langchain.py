from typing import Any, Optional
from PIL.Image import Image
import os

from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

import dotenv


"""
See https://python.langchain.com/docs/how_to/qa_sources/
"""
model_providers = {
    "google_vertexai": {"gemini-2.0-flash-001"},
    "google_genai": {"gemini-2.5-flash"},
    "openai": {"gpt-4o"},
}

# https://python.langchain.com/docs/integrations/chat/
# need to first install deps using 
#       pip install -qU "langchain[provider]"
def create_model(provider: Optional[str], 
                      model: Optional[str]) -> BaseChatModel: 
    print("loading .env")
    print(os.getcwd())
    dotenv.load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

    if provider is None or model is None: # default
        provider = "openai"
        model = "gpt-4o"
        
    if (provider not in model_providers) or \
       (provider in model_providers and model not in model_providers[provider]):
        raise ValueError(
            f"Invalid model or provider. Available provider and models: {model_providers.keys()}"
        )
        
    chat_model = init_chat_model(model, model_provider=provider)
    return chat_model


# https://python.langchain.com/docs/tutorials/qa_chat_history/
class TextGenerationModel:
    def __init__(self, *, provider: Optional[str]=None, model: Optional[str]=None, system_message: Optional[str] = ""):
        self.provider = provider
        self.model = model
        self._system_message = system_message
        self.llm = create_model(provider, model)
        
    @property
    def system_message(self) -> str:
        return self._system_message
    
    @system_message.setter
    def system_message(self, msg: str):
        self._system_message = msg

    def generate(self, question: str, context: str = "", *, images: Optional[list[Any]] = None) -> BaseMessage:
        if images is not None:
            img_content = [
                    {
                        "type": "text",
                        "text": question,
                    },
                    ]
            img_content.extend(
                { # for base64 enc. imgs
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}"
                    }
                } for img in images
                    # {
                    #     "type": "image",
                    #     "source_type": "base64",
                    #     "data": images,
                    #     "mime_type": "image/jpeg",
                    # },
            )
            message = {
                "role": "user",
                "content": img_content
            }
        else:
            message = question

        prompt = [SystemMessage(self.system_message + context)]  + [message]
        response = self.llm.invoke(prompt)
        return response.content