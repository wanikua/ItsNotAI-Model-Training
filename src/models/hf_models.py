import torch
from torchvision import transforms
from PIL.Image import Image as PILImage
from transformers import (
    AutoFeatureExtractor, AutoModelForImageClassification, AutoImageProcessor, pipeline,
    SiglipForImageClassification
)
from timm import create_model
from huggingface_hub import hf_hub_download

from src.models.model_api import HuggingFaceModel, HfModelOutput
from src.utils.model_utils import sanitize_label, get_device, BatchableMixin
    

class AIOrNotHfModel(HuggingFaceModel):
    """
    https://huggingface.co/Nahrawy/AIorNot
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["Real", "AI"]
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("Nahrawy/AIorNot")
        self.model = AutoModelForImageClassification.from_pretrained("Nahrawy/AIorNot")
        self.device = get_device()
        print(f"[{self.__class__.__name__}] Model initialization done.")

    def predict(self, img: PILImage, *, with_probs: bool = True) -> HfModelOutput:
        inputs = self.feature_extractor(img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = sanitize_label([self.labels[prediction]])[0]
            
            # Always calculate probabilities for unified interface
            probs = torch.softmax(logits, dim=-1)[-1].tolist()

        return HfModelOutput(label=label, probs=probs)


class SDXLDetectorHfModel(HuggingFaceModel):
    """
        https://colab.research.google.com/#fileId=https%3A//huggingface.co/Organika/sdxl-detector.ipynb
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = ["AI", "Real"]
        self.processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
        self.model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
        self.device = get_device()
        print(f"[{self.__class__.__name__}] Model initialization done.")

    def predict(self, img: PILImage, *, with_probs: bool = True) -> HfModelOutput:
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = logits.argmax(-1).item()
            label = sanitize_label([self.labels[prediction]])[0]
            
            # Always calculate probabilities for unified interface
            probs = torch.softmax(logits, dim=-1)[-1].tolist() 
                
        return HfModelOutput(label=label, probs=probs)
    
    
class AIVSHumanImageDetectorHfModel(HuggingFaceModel):
    """ 
        https://huggingface.co/Ateeqq/ai-vs-human-image-detector
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_identifier = r"Ateeqq/ai-vs-human-image-detector"
        self.device = get_device()
        
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
    

    def predict(self, img: PILImage, *, with_probs: bool = True) -> HfModelOutput:
        inputs = self.processor(img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction_idx = logits.argmax(-1).item()
            label = sanitize_label([self.model.config.id2label[prediction_idx]])[0]
            
            # Always calculate probabilities for unified interface
            probs = torch.softmax(logits, dim=-1)[-1].tolist()

            return HfModelOutput(label=label, probs=probs)
        
        
class DafilabAIImageDetectorHfModel(HuggingFaceModel):
    """
        https://huggingface.co/Dafilab/ai-image-detector 
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
        # Parameters
        IMG_SIZE = 380
        self.device = get_device()
        self.label_mapping = {1: "human", 0: "ai"}

        # Download model from HuggingFace Hub
        MODEL_PATH = hf_hub_download(repo_id="Dafilab/ai-image-detector", filename="pytorch_model.pth")

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE + 20),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load model
        self.model = create_model('efficientnet_b4', pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device).eval()
        
        print(f"[{self.__class__.__name__}] Model initialization done.")


    def predict(self, img: PILImage, *, with_probs: bool = True) -> HfModelOutput:
        img = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img)
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            label = sanitize_label([self.label_mapping[predicted_class]])[0]
            
            # Always calculate probabilities for unified interface
            probs_list = probs[0].tolist()

            return HfModelOutput(label=label, probs=probs_list)
    
        
    
       




  