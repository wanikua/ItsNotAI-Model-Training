"""
A generric wrapper for all models
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Optional, List

class DiscriminatorModel(ABC):
    """
    A generic wrapper for all models
    """
    @abstractmethod
    def __init__(self, model_name: str, *args, **kwargs):
        self.name = model_name

    @abstractmethod
    def predict(self, input: Any) -> Any:
        pass


class OpenSourceModel(DiscriminatorModel):
    """
    A generic wrapper for all open source models
    """
    def __init__(self, *args, **kwargs):
        super().__init__(self.__class__.__name__, *args, **kwargs)
        self.platform = ""


class HuggingFaceModel(OpenSourceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self.__class__.__name__, *args, **kwargs)
        self.platform = "huggingface"


class GitHubModel(OpenSourceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self.__class__.__name__, *args, **kwargs)
        self.platform = "github"


class APIProvider(DiscriminatorModel):
    def __init__(self, *args, **kwargs):
        super().__init__(self.__class__.__name__, *args, **kwargs)  
        self.provider = ""
        

class HfModelOutput(BaseModel):
    label: str          
    probs: Optional[List[float]]
