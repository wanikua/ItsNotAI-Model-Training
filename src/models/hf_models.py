import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor, pipeline

from src.models.model_api import HuggingFaceModel




class AIOrNotHfModel(HuggingFaceModel):
    """
    https://huggingface.co/Nahrawy/AIorNot
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["Real", "AI"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Nahrawy/AIorNot")
        self.model = AutoModelForImageClassification.from_pretrained("Nahrawy/AIorNot")

    def predict(self, img: torch.Tensor) -> str:
        input = self.feature_extractor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**input)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = self.labels[prediction] 

        return label


class SDXLDetectorHfModel(HuggingFaceModel):
    """
        https://colab.research.google.com/#fileId=https%3A//huggingface.co/Organika/sdxl-detector.ipynb
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["Real", "AI"]
        self.processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
        self.model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

    def predict(self, img: torch.Tensor) -> str:
        input = self.processor(img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**input)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = self.labels[prediction] 

        return label

    
       




  