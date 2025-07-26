from pydantic import BaseModel
from typing import List, Optional

import torch
from PIL import Image as PILImage
from transformers import (
    AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor, pipeline,
    SiglipForImageClassification
)

from src.models.model_api import HuggingFaceModel
from src.utils.model_utils import sanitize_label

class HfModelOutput(BaseModel):
    label: str          
    probs: Optional[List[float]]
    

class AIOrNotHfModel(HuggingFaceModel):
    """
    https://huggingface.co/Nahrawy/AIorNot
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["Real", "AI"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Nahrawy/AIorNot")
        self.model = AutoModelForImageClassification.from_pretrained("Nahrawy/AIorNot")
        print(f"[{self.__class__.__name__}] Model initialization done.")

    def predict(self, img: torch.Tensor, *, with_probs: bool = False) -> str:
        inputs = self.feature_extractor(img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = sanitize_label(self.labels[prediction])

        if with_probs:
            probs = torch.softmax(logits, dim=-1)[-1].tolist()
            return HfModelOutput(label=label, probs=probs)
                
        return HfModelOutput(label=label)


class SDXLDetectorHfModel(HuggingFaceModel):
    """
        https://colab.research.google.com/#fileId=https%3A//huggingface.co/Organika/sdxl-detector.ipynb
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["Real", "AI"]
        self.processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
        self.model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
        print(f"[{self.__class__.__name__}] Model initialization done.")

    def predict(self, img: PILImage, *, with_probs: bool = False) -> str:
        inputs = self.processor(img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = sanitize_label(self.labels[prediction])
            
            if with_probs:
                probs = torch.softmax(logits, dim=-1)[-1].tolist()
                return HfModelOutput(label=label, probs=probs)
                
        return HfModelOutput(label=label)
    
    
class AIVSHumanImageDetectorHfModel(HuggingFaceModel):
    """ 
        https://huggingface.co/Ateeqq/ai-vs-human-image-detector
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_identifier = r"Ateeqq/ai-vs-human-image-detector"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model and Processor
        try:
            print(f"Loading processor from: {self.model_identifier}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_identifier)

            print(f"Loading model from: {self.model_identifier}")
            self.model = SiglipForImageClassification.from_pretrained(self.model_identifier)
            self.model.to(self.device)
            self.model.eval()
            print("Model and processor loaded successfully.")

        except Exception as e:
            print(f"Error loading model or processor: {e}")
            exit()
            
        print(f"[{self.__class__.__name__}] Model initialization done.")
    

    def predict(self, img: PILImage, *, with_probs: bool = False) -> str:
        inputs = self.processor(img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction_idx = logits.argmax(-1).item()
            label = sanitize_label(self.model.config.id2label[prediction_idx])
            
            if with_probs:
                probs = torch.softmax(logits, dim=-1)[-1].tolist()
                # predicted_prob = probabilities[0, predicted_class_idx].item()
                return HfModelOutput(label=label, probs=probs)

            return HfModelOutput(label=label)
        
    
       




  