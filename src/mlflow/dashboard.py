# # dashboard.py
# import mlflow
# import pandas as pd
# import streamlit as st

# st.title("AI Image Detection Dashboard")

# # Get runs
# client = mlflow.tracking.MlflowClient()
# experiment = client.get_experiment_by_name("AI Image Detector v0")
# runs = client.search_runs([experiment.experiment_id])

# df = pd.DataFrame([{**r.data.metrics, **r.data.params} for r in runs])
# st.dataframe(df)
# st.line_chart(df["train_accuracy"])


# app.py
import os
import io
import glob
import random
import time
import json
import base64
from pathlib import Path
from typing import Optional, Union, Any, Dict, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import cv2
import pandas as pd

# Import VLM model
try:
    from src.models.langchain import TextGenerationModel, model_providers
    VLM_AVAILABLE = True
except ImportError as e:
    st.warning(f"VLM functionality not available: {e}")
    VLM_AVAILABLE = False

# Import API providers and HF models
try:
    from src.models.api_providers import HiveAPIProvider, SightengineAPIProvider, OpenAI
    from src.models.hf_models import AIOrNotHfModel, SDXLDetectorHfModel, AIVSHumanImageDetectorHfModel, DafilabAIImageDetectorHfModel
    EXTERNAL_MODELS_AVAILABLE = True
except ImportError as e:
    st.warning(f"External models functionality not available: {e}")
    EXTERNAL_MODELS_AVAILABLE = False

# Import model utilities
from src.utils.model_utils import sanitize_label

# Avoid importing torch at module level to prevent Streamlit watcher issues
# Import torch only when needed inside functions

st.set_page_config(page_title="AI Image Detector Demo", layout="wide")

# Configuration with better defaults
DATASET_ROOT = os.environ.get("DATASET_ROOT", "data/test/archive")  # <- real/ and fake-v2/
if not os.path.exists(DATASET_ROOT):
    st.error(f"Dataset root '{DATASET_ROOT}' does not exist. Please check the path.")
    st.stop()

IMAGE_SUBTYPES = ["photography", "news", "cartoon", "art", "render", "meme"]  # placeholders

def get_model_transform():
    """Get the model preprocessing transform (imports torch only when needed)."""
    try:
        # Import only when needed to avoid Streamlit watcher issues
        import torch
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    except ImportError as e:
        st.warning(f"PyTorch/torchvision not available: {e}. Using fallback preprocessing.")
        return None
    except Exception as e:
        st.error(f"Failed to create image transforms: {e}")
        return None

def preprocess_image_fallback(img: Image.Image) -> np.ndarray:
    """Fallback image preprocessing without torch."""
    # Resize to 224x224
    img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert from HWC to CHW format (channels first)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Label mapping to match training
LABEL_MAP = {0: "real", 1: "fake"}
INV_LABEL_MAP = {"real": 0, "fake": 1}

# ---------- Helpers ----------
def list_models() -> List[Dict[str, str]]:
    """List available MLflow models with error handling."""
    try:
        client = MlflowClient()
        out = []
        for rm in client.search_registered_models():
            print("found registered model:", rm.name)
            # prefer Production, else latest
            mv = None
            for v in rm.latest_versions:
                if v.current_stage == "Production":
                    mv = v
                    break
            if not mv and rm.latest_versions:
                mv = max(rm.latest_versions, key=lambda x: int(x.version))
            if not mv:
                continue
            out.append({
                "name": rm.name,
                "label": f"{rm.name} ({mv.current_stage or 'v'+mv.version})",
                "uri": f"models:/{rm.name}/{mv.current_stage or mv.version}",
                "type": "mlflow"
            })
        return out
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return []

def list_api_providers() -> List[Dict[str, str]]:
    """List available API providers with their required keys."""
    if not EXTERNAL_MODELS_AVAILABLE:
        return []
    
    providers = [
        {
            "name": "HiveAPI",
            "label": "Hive AI Detection API",
            "class": HiveAPIProvider,
            "required_keys": ["HIVE_API_KEY"],
            "type": "api"
        },
        {
            "name": "Sightengine",
            "label": "Sightengine AI Detection API", 
            "class": SightengineAPIProvider,
            "required_keys": ["SIGHTENGINE_API_USER", "SIGHTENGINE_API_SECRET"],
            "type": "api"
        },
        {
            "name": "OpenAI",
            "label": "OpenAI API (Custom)",
            "class": OpenAI,
            "required_keys": ["OPENAI_API_KEY"],
            "type": "api"
        }
    ]
    return providers

def list_hf_models() -> List[Dict[str, str]]:
    """List available Hugging Face models."""
    if not EXTERNAL_MODELS_AVAILABLE:
        return []
    
    models = [
        {
            "name": "AIOrNot",
            "label": "AI or Not (Nahrawy/AIorNot)",
            "class": AIOrNotHfModel,
            "type": "huggingface"
        },
        {
            "name": "SDXL-Detector", 
            "label": "SDXL Detector (Organika/sdxl-detector)",
            "class": SDXLDetectorHfModel,
            "type": "huggingface"
        },
        {
            "name": "AI-vs-Human",
            "label": "AI vs Human Detector (Ateeqq)",
            "class": AIVSHumanImageDetectorHfModel,
            "type": "huggingface"
        },
        {
            "name": "Dafilab-Detector",
            "label": "Dafilab AI Image Detector",
            "class": DafilabAIImageDetectorHfModel,
            "type": "huggingface"
        }
    ]
    return models

def check_api_keys(required_keys: List[str]) -> tuple[bool, List[str]]:
    """Check if required API keys are available."""
    missing_keys = []
    for key in required_keys:
        if not os.environ.get(key):
            missing_keys.append(key)
    return len(missing_keys) == 0, missing_keys

def get_all_models() -> List[Dict[str, str]]:
    """Get all available models from all sources."""
    all_models = []
    
    # MLflow models
    mlflow_models = list_models()
    all_models.extend(mlflow_models)
    
    # API providers
    api_providers = list_api_providers()
    all_models.extend(api_providers)
    
    # HuggingFace models
    hf_models = list_hf_models()
    all_models.extend(hf_models)
    
    return all_models

@st.cache_resource
def load_model(uri: str):
    """Load MLflow model with error handling."""
    try:
        return mlflow.pyfunc.load_model(uri)
    except Exception as e:
        st.error(f"Error loading model from {uri}: {str(e)}")
        return None

def sample_random_image() -> Optional[str]:
    """Sample a random image from the dataset."""
    try:
        paths = []
        for sub in ("real", "fake-v2"):
            sub_path = Path(DATASET_ROOT) / sub
            if not sub_path.exists():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                paths.extend(glob.glob(str(sub_path / ext)))
        return random.choice(paths) if paths else None
    except Exception as e:
        st.error(f"Error sampling image: {str(e)}")
        return None

def sample_random_image_from_category(category: str) -> Optional[str]:
    """Sample a random image from a specific category (real or fake)."""
    try:
        if category == "real":
            sub_folder = "real"
        elif category == "fake":
            sub_folder = "fake-v2"
        else:
            return sample_random_image()
        
        paths = []
        sub_path = Path(DATASET_ROOT) / sub_folder
        if sub_path.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                paths.extend(glob.glob(str(sub_path / ext)))
        return random.choice(paths) if paths else None
    except Exception as e:
        st.error(f"Error sampling image from {category}: {str(e)}")
        return None

def load_image(path: str, max_w: int = 900) -> Optional[Image.Image]:
    """Load and resize image with error handling."""
    try:
        if not os.path.exists(path):
            st.error(f"Image file not found: {path}")
            return None
        
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            h = int(im.height * max_w / im.width)
            im = im.resize((max_w, h), Image.Resampling.LANCZOS)
        return im
    except Exception as e:
        st.error(f"Error loading image {path}: {str(e)}")
        return None


def overlay_tag(im: Image.Image, text: str) -> Image.Image:
    """Overlay text tag on image with robust font handling."""
    if im is None:
        return None
    
    im = im.copy()
    draw = ImageDraw.Draw(im)
    
    # Try multiple font options with fallbacks
    font = None
    font_size = 20
    font_options = [
        "DejaVuSans-Bold.ttf",
        "Arial-Bold.ttf", 
        "Helvetica-Bold.ttf",
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
    ]
    
    for font_path in font_options:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    
    # Fallback to default font
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            # If all else fails, skip text overlay
            return im
    
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 10
        x, y = 10, im.height - th - pad * 2 - 10
        draw.rounded_rectangle(
            [x, y, x + tw + pad * 2, y + th + pad * 2], 
            radius=10, 
            fill=(20, 20, 20, 230)
        )
        draw.text((x + pad, y + pad), text, font=font, fill=(255, 255, 255))
    except Exception as e:
        # If text overlay fails, just return the original image
        st.warning(f"Could not overlay text: {str(e)}")
    
    return im


def predict_image(model, path: str) -> Any:
    """Predict using MLflow model."""
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    import io
    import base64
    
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        # Load and preprocess image
        img = Image.open(path).convert("RGB")
        
        # Define transform (adjust based on your model's requirements)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transform and add batch dimension
        img_tensor = transform(img).unsqueeze(0)
        
        # Predict using MLflow model's predict method
        with torch.no_grad():
            output = model.predict(img_tensor.numpy())  # MLflow models expect numpy arrays
            
        return output
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        raise

def load_external_model(model_info: Dict[str, str]):
    """Load external model (API or HuggingFace)."""
    try:
        model_class = model_info["class"]
        if model_info["type"] == "api":
            # For API models, just return the class - instantiation happens during prediction
            return model_class
        elif model_info["type"] == "huggingface":
            # For HF models, instantiate the class
            return model_class()
        return None
    except Exception as e:
        st.error(f"Error loading {model_info['name']}: {str(e)}")
        return None


def predict_with_external_model(model_info: Dict[str, str], image_path: str):
    """Predict using external models (API or HuggingFace)."""
    try:
        if model_info["type"] == "api":
            # For API models, instantiate with proper credentials
            model_class = model_info["class"]
            if model_info["name"] == "HiveAPI":
                api_key = os.environ.get("HIVE_API_KEY")
                model = model_class(api_key=api_key)
            elif model_info["name"] == "Sightengine":
                api_user = os.environ.get("SIGHTENGINE_API_USER")
                api_secret = os.environ.get("SIGHTENGINE_API_SECRET")
                model = model_class(api_user=api_user, api_secret=api_secret)
            elif model_info["name"] == "OpenAI":
                api_key = os.environ.get("OPENAI_API_KEY")
                model = model_class(api_key=api_key)
            else:
                st.error(f"Unknown API provider: {model_info['name']}")
                return None
            
            # API models typically expect file paths
            return model.predict(image_path)
            
        elif model_info["type"] == "huggingface":
            # For HF models, load the image as PIL Image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load image as PIL Image (what HF models expect)
            image = Image.open(image_path).convert("RGB")
            
            # Load and predict with HF model
            model = load_external_model(model_info)
            if model:
                return model.predict(image).dict()  # Pass PIL Image instead of path
            return None
    except Exception as e:
        st.error(f"Error predicting with {model_info['name']}: {str(e)}")
        return None


def unified_predict(model_info: Dict[str, str], image_path: str, model_uri: str = None) -> Dict[str, Any]:
    """
    Unified prediction function that handles all model types and standardizes output format.
    
    Returns:
        Dict with keys: 'raw', 'confidence', 'predicted_class', 'probabilities', 'model_type'
    """
    try:
        if model_info.get("type") == "mlflow":
            # MLflow model prediction
            model = load_model(model_uri)
            if model is None:
                return {"error": "Failed to load MLflow model", "model_type": "mlflow"}
            
            raw_output = predict_image(model, image_path)
            
            # Convert MLflow output to standard format
            if isinstance(raw_output, (list, tuple, np.ndarray)):
                if hasattr(raw_output, '__len__') and len(raw_output) > 0:
                    if isinstance(raw_output[0], (list, np.ndarray)):
                        # Handle 2D output (batch, classes)
                        probs = raw_output[0] if len(raw_output) > 0 else raw_output
                    else:
                        probs = raw_output
                    
                    # Convert to numpy if needed
                    if not isinstance(probs, np.ndarray):
                        probs = np.array(probs)
                    
                    # Apply softmax if raw logits
                    if np.any(probs < 0) or np.sum(probs) > 1.1:
                        try:
                            try:
                                from scipy.special import softmax
                                probs = softmax(probs)
                            except ImportError:
                                # Fallback manual softmax implementation
                                exp_probs = np.exp(probs - np.max(probs))
                                probs = exp_probs / np.sum(exp_probs)
                        except Exception:
                            # If softmax fails, normalize manually
                            probs = probs / np.sum(probs) if np.sum(probs) > 0 else probs
                    
                    confidence = float(np.max(probs))
                    predicted_class = "fake" if np.argmax(probs) == 1 else "real"
                    
                    return {
                        "raw": raw_output,
                        "confidence": confidence,
                        "predicted_class": predicted_class,
                        "probabilities": {"real": float(probs[0]), "fake": float(probs[1]) if len(probs) > 1 else 1-float(probs[0])},
                        "model_type": "mlflow"
                    }
            
            return {"error": f"Unexpected MLflow output format: {type(raw_output)}", "model_type": "mlflow"}
            
        else:
            # External model prediction (API or HuggingFace)
            raw_output = predict_with_external_model(model_info, image_path)
            
            if raw_output is None:
                return {"error": "External model prediction failed", "model_type": model_info["type"]}
            
            # Standardize external model outputs
            if isinstance(raw_output, dict): # TODO: check with BaseModel Model API instead of dict
                if 'confidence' in raw_output and 'prediction' in raw_output:
                    return {
                        "raw": raw_output,
                        "confidence": float(raw_output['confidence']),
                        "predicted_class": raw_output['prediction'],
                        "probabilities": raw_output.get('probabilities', {}),
                        "model_type": model_info["type"]
                    }
                elif 'ai_probability' in raw_output:
                    ai_prob = float(raw_output['ai_probability'])
                    return {
                        "raw": raw_output,
                        "confidence": max(ai_prob, 1-ai_prob),
                        "predicted_class": "AI-generated" if ai_prob > 0.5 else "Real",
                        "probabilities": {"Real": 1-ai_prob, "AI-generated": ai_prob},
                        "model_type": model_info["type"]
                    }
                elif 'label' in raw_output and 'probs' in raw_output: # HfModels API
                    label = raw_output['label']
                    probs = raw_output['probs']
                    confidence = max(probs) if probs else 0.8
                    return {
                        "raw": raw_output,
                        "confidence": confidence,
                        "predicted_class": label,
                        "probabilities": {
                            "real": float(probs[0]),
                            "fake": float(probs[1])
                        } if probs else {},
                        "model_type": model_info["type"]
                    }
                    
            return {
                "raw": raw_output,
                "confidence": 0.5,
                "predicted_class": "Unknown",
                "probabilities": {"Real": 0.5, "AI-generated": 0.5},
                "model_type": model_info["type"]
            }
            
    except Exception as e:
        return {
            "error": f"Prediction failed: {str(e)}",
            "model_type": model_info.get("type", "unknown")
        }

def run_model_prediction(model, path: str):
    """Run model prediction with proper preprocessing to match training pipeline."""
    if model is None:
        raise ValueError("Model is None")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        # Load and preprocess image exactly like training
        img = Image.open(path).convert("RGB")
        
        # Try to get the torch-based transform first
        model_transform = get_model_transform()
        
        if model_transform is not None:
            # Use torch transforms if available
            try:
                tensor_input = model_transform(img).unsqueeze(0)  # Add batch dimension
                tensor_numpy = tensor_input.numpy()
            except Exception as e:
                st.warning(f"Torch transform failed: {e}. Using fallback.")
                tensor_numpy = preprocess_image_fallback(img)
        else:
            # Use fallback preprocessing
            tensor_numpy = preprocess_image_fallback(img)
        
        # Try different input formats based on MLflow model signature
        payloads = [
            tensor_numpy,  # Tensor as numpy array (most likely)
            {"image": tensor_numpy},  # Dictionary format
            pd.DataFrame([{"image": tensor_numpy}]),  # DataFrame format
        ]
        
        # If torch is available, also try raw tensor
        if model_transform is not None:
            try:
                import torch
                tensor_input = torch.from_numpy(tensor_numpy)
                payloads.insert(1, tensor_input)  # Raw tensor (if PyTorch model)
            except ImportError:
                pass
        
        last_error = None
        for i, payload in enumerate(payloads):
            try:
                result = model.predict(payload)
                return result
            except Exception as e:
                last_error = e
                continue
        
        # If all payloads failed, raise the last error with more context
        raise RuntimeError(
            f"Model prediction failed with all payload formats. "
            f"Last error: {str(last_error)}. "
            f"Expected input: tensor of shape (1, 3, 224, 224). "
            f"Tried {len(payloads)} different payload formats."
        )
        
    except Exception as e:
        if "not found" in str(e).lower():
            raise FileNotFoundError(f"Image file not found: {path}")
        else:
            raise RuntimeError(f"Error during prediction: {str(e)}")


def interpret_prediction(prediction_result) -> Dict[str, Any]:
    """Interpret prediction results from various model types."""
    import numpy as np
    
    try:
        # Handle None or empty results
        if prediction_result is None:
            return {"predicted_class": "error", "confidence": 0.0, "error": "No prediction result"}
        
        # Handle HfModelOutput objects (from our HuggingFace models)
        if hasattr(prediction_result, 'label') and hasattr(prediction_result, 'probs'):
            label = prediction_result.label
            probs = prediction_result.probs
            
            # Use sanitize_label to normalize the label
            try:
                normalized_label = sanitize_label([label])[0]
            except ValueError:
                # If sanitize_label fails, try manual mapping
                if "fake" in label.lower() or "ai" in label.lower() or "artificial" in label.lower():
                    normalized_label = "fake"
                else:
                    normalized_label = "real"
            
            pred_class = normalized_label
            pred_id = 1 if pred_class == "fake" else 0
            
            # Use probabilities if available
            if probs and len(probs) >= 2:
                confidence = max(probs)
                return {
                    "predicted_class": pred_class,
                    "predicted_class_id": pred_id,
                    "confidence": confidence,
                    "probabilities": {
                        "real": float(probs[0] if pred_class == "real" else probs[1]),
                        "fake": float(probs[1] if pred_class == "fake" else probs[0])
                    },
                    "raw_output": {"label": label, "probs": probs}
                }
            else:
                return {
                    "predicted_class": pred_class,
                    "predicted_class_id": pred_id,
                    "confidence": 0.8,
                    "raw_output": {"label": label}
                }
        
        # Handle string responses (API models often return text)
        if isinstance(prediction_result, str):
            # Try to use sanitize_label first
            try:
                normalized_label = sanitize_label([prediction_result])[0]
                pred_class = normalized_label
                pred_id = 1 if pred_class == "fake" else 0
                return {
                    "predicted_class": pred_class,
                    "predicted_class_id": pred_id,
                    "confidence": 0.8,
                    "raw_output": prediction_result
                }
            except ValueError:
                # If sanitize_label fails, try manual parsing
                lower_result = prediction_result.lower()
                if "ai" in lower_result or "artificial" in lower_result or "generated" in lower_result or "fake" in lower_result:
                    return {
                        "predicted_class": "fake",
                        "predicted_class_id": 1,
                        "confidence": 0.8,
                        "raw_output": prediction_result
                    }
                elif "real" in lower_result or "human" in lower_result or "authentic" in lower_result:
                    return {
                        "predicted_class": "real", 
                        "predicted_class_id": 0,
                        "confidence": 0.8,
                        "raw_output": prediction_result
                    }
                else:
                    return {
                        "predicted_class": "unknown",
                        "confidence": 0.0,
                        "raw_output": prediction_result
                    }
        
        # Handle dictionary responses (common from API providers and HF models)
        if isinstance(prediction_result, dict):
            # Handle HfModelOutput format
            if "label" in prediction_result and "probs" in prediction_result:
                label = prediction_result["label"]
                probs = prediction_result["probs"]
                
                # Use sanitize_label to normalize the label
                try:
                    normalized_label = sanitize_label([label])[0]
                except ValueError:
                    # If sanitize_label fails, try manual mapping
                    if "fake" in label.lower() or "ai" in label.lower() or "artificial" in label.lower():
                        normalized_label = "fake"
                    else:
                        normalized_label = "real"
                
                pred_class = normalized_label
                pred_id = 1 if pred_class == "fake" else 0
                
                # If we have probs, use them; otherwise use default confidence
                if probs and len(probs) >= 2:
                    confidence = max(probs)
                    return {
                        "predicted_class": pred_class,
                        "predicted_class_id": pred_id,
                        "confidence": confidence,
                        "probabilities": {
                            "real": float(probs[0] if pred_class == "real" else probs[1]),
                            "fake": float(probs[1] if pred_class == "fake" else probs[0])
                        },
                        "raw_output": prediction_result
                    }
                else:
                    return {
                        "predicted_class": pred_class,
                        "predicted_class_id": pred_id,
                        "confidence": 0.8,
                        "raw_output": prediction_result
                    }
            
            # Handle common API response formats
            if "classes" in prediction_result and "scores" in prediction_result:
                # HuggingFace-style output
                classes = prediction_result["classes"]
                scores = prediction_result["scores"]
                max_idx = np.argmax(scores)
                predicted_label = classes[max_idx]
                confidence = float(scores[max_idx])
                
                # Use sanitize_label to normalize the label
                try:
                    normalized_label = sanitize_label([predicted_label])[0]
                except ValueError:
                    # If sanitize_label fails, try manual mapping
                    if "fake" in predicted_label.lower() or "ai" in predicted_label.lower():
                        normalized_label = "fake"
                    else:
                        normalized_label = "real"
                
                pred_class = normalized_label
                pred_id = 1 if pred_class == "fake" else 0
                
                return {
                    "predicted_class": pred_class,
                    "predicted_class_id": pred_id,
                    "confidence": confidence,
                    "raw_output": prediction_result
                }
            
            # Handle other dict formats
            for key in ["prediction", "result", "class", "label"]:
                if key in prediction_result:
                    value = prediction_result[key]
                    if isinstance(value, str):
                        try:
                            normalized_label = sanitize_label([value])[0]
                            pred_class = normalized_label
                            pred_id = 1 if pred_class == "fake" else 0
                            return {"predicted_class": pred_class, "predicted_class_id": pred_id, "confidence": 0.8, "raw_output": prediction_result}
                        except ValueError:
                            # Fallback to manual mapping
                            if "fake" in value.lower() or "ai" in value.lower():
                                return {"predicted_class": "fake", "predicted_class_id": 1, "confidence": 0.8, "raw_output": prediction_result}
                            else:
                                return {"predicted_class": "real", "predicted_class_id": 0, "confidence": 0.8, "raw_output": prediction_result}
        
        # Handle numpy arrays (MLflow models)
        if isinstance(prediction_result, np.ndarray):
            if prediction_result.ndim == 2 and prediction_result.shape[1] == 2:
                # Softmax output: shape (1, 2) with probabilities [real_prob, fake_prob]
                probs = prediction_result[0]
                predicted_class = np.argmax(probs)
                confidence = float(probs[predicted_class])
                
                return {
                    "predicted_class": LABEL_MAP[predicted_class],
                    "predicted_class_id": int(predicted_class),
                    "confidence": confidence,
                    "probabilities": {
                        "real": float(probs[0]),
                        "fake": float(probs[1])
                    },
                    "raw_output": probs.tolist()
                }
            elif prediction_result.ndim == 1 and len(prediction_result) == 2:
                # Direct probability output
                probs = prediction_result
                predicted_class = np.argmax(probs)
                confidence = float(probs[predicted_class])
                
                return {
                    "predicted_class": LABEL_MAP[predicted_class],
                    "predicted_class_id": int(predicted_class),
                    "confidence": confidence,
                    "probabilities": {
                        "real": float(probs[0]),
                        "fake": float(probs[1])
                    },
                    "raw_output": probs.tolist()
                }
            else:
                # Unknown format, return as-is
                return {
                    "predicted_class": "unknown",
                    "confidence": 0.0,
                    "raw_output": prediction_result.tolist(),
                    "note": f"Unexpected output shape: {prediction_result.shape}"
                }
        else:
            # Non-array result
            return {
                "predicted_class": "unknown",
                "confidence": 0.0,
                "raw_output": str(prediction_result),
                "note": f"Unexpected output type: {type(prediction_result)}"
            }
    except Exception as e:
        return {
            "predicted_class": "error",
            "confidence": 0.0,
            "raw_output": str(prediction_result),
            "error": str(e)
        }

def spectrum_image(im: Image.Image, size: int = 512) -> Optional[Image.Image]:
    """Generate frequency spectrum visualization with error handling."""
    if im is None:
        return None
    
    try:
        gray = np.array(im.convert("L").resize((size, size), Image.Resampling.LANCZOS))
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        mag = np.log(mag + 1.0)
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return Image.fromarray(mag)
    except Exception as e:
        st.error(f"Error generating spectrum image: {str(e)}")
        return None

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def test_vlm_model(provider: str, model: str) -> tuple[bool, str]:
    """Test if VLM model is available with given provider and model."""
    try:
        vlm = TextGenerationModel(provider=provider, model=model)
        # Test with a simple prompt
        test_response = vlm.generate("Hello", "")
        return True, "Model available"
    except Exception as e:
        return False, str(e)

def vlm_analysis(image: Image.Image, subtype: str, provider: str, model: str) -> str:
    """Generate VLM analysis of the image."""
    try:
        # Convert image to base64
        img_base64 = image_to_base64(image)
        
        # Create VLM model
        vlm = TextGenerationModel(
            provider=provider, 
            model=model,
            system_message="""You are an expert AI image analyst specializing in detecting AI-generated vs real images. 
            Analyze the provided image carefully and provide detailed technical analysis about whether it appears to be AI-generated or real."""
        )
        
        # Create analysis prompt
        prompt = f"""Analyze this image (subtype: {subtype}) for signs of AI generation vs real photography. 
        
        Consider these aspects:
        1. Texture consistency and artifacts
        2. Lighting and shadow realism
        3. Fine details and edge quality
        4. Color saturation and distribution
        5. Facial features (if present) and anatomical correctness
        6. Background consistency
        7. Compression artifacts vs generation artifacts
        
        Provide your analysis in the following format:
        **Analysis for {subtype}:**
        - [Key observation 1] 
        - [Key observation 2] 
        - [Key observation 3] 

        **Verdict:** [Real/AI-Generated] with confidence level
        **Reasoning:** [Brief explanation of key factors]
        """
        
        # Generate analysis
        response = vlm.generate(prompt, images=[img_base64])
        return response
        
    except Exception as e:
        return f"Error during VLM analysis: {str(e)}"

# ---------- UI State ----------
ss = st.session_state
for k in ("candidate", "selected", "pred", "compare"):
    ss.setdefault(k, None)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)
    st.header("🤖 Model Zoo")
    
    # Model type selection
    model_type = st.selectbox(
        "Model Type", 
        ["MLflow Models", "API Providers", "HuggingFace Models"],
        help="Choose the type of model to use for detection"
    )
    
    selected_model = None
    model_uri = None
    
    if model_type == "MLflow Models":
        models = list_models()
        if not models:
            st.error("No MLflow models found!")
        else:
            labels = [m["label"] for m in models]
            model_names = [m["name"] for m in models]
            model_uris = [m["uri"] for m in models]
            selected_idx = st.selectbox("Select Model", range(len(labels)), format_func=lambda i: labels[i])
            model_uri = model_uris[selected_idx]
            selected_model = models[selected_idx]
            st.success(f"Selected: {model_names[selected_idx]}")
    
    elif model_type == "API Providers":
        if not EXTERNAL_MODELS_AVAILABLE:
            st.error("External model dependencies not available. Please install requirements.")
        else:
            api_providers = list_api_providers()
            if api_providers:
                labels = [p["label"] for p in api_providers]
                selected_idx = st.selectbox("Select API Provider", range(len(labels)), format_func=lambda i: labels[i])
                selected_provider = api_providers[selected_idx]
                selected_model = selected_provider
                
                # Check API keys
                has_keys, missing_keys = check_api_keys(selected_provider["required_keys"])
                if has_keys:
                    st.success(f"Selected: {selected_provider['name']}")
                else:
                    st.error(f"Missing API keys: {', '.join(missing_keys)}")
                    selected_model = None
    
    elif model_type == "HuggingFace Models":
        if not EXTERNAL_MODELS_AVAILABLE:
            st.error("External model dependencies not available. Please install requirements.")
        else:
            hf_models = list_hf_models()
            if hf_models:
                labels = [m["label"] for m in hf_models]
                selected_idx = st.selectbox("Select HF Model", range(len(labels)), format_func=lambda i: labels[i])
                selected_model = hf_models[selected_idx]
                st.success(f"Selected: {selected_model['name']}")
    
    st.markdown('<hr style="margin: 0;">', unsafe_allow_html=True)
    st.header("🔧 VLM Analysis")
    
    # Store VLM enabled state in session state
    if 'vlm_enabled' not in ss:
        ss.vlm_enabled = True
    
    ss.vlm_enabled = st.checkbox("Enable VLM Analysis", value=ss.vlm_enabled)
    
    # Initialize VLM configuration in session state
    if 'vlm_provider' not in ss:
        ss.vlm_provider = None
    if 'vlm_model' not in ss:
        ss.vlm_model = None
    
    if ss.vlm_enabled:
        if not VLM_AVAILABLE:
            st.error("VLM functionality is not available. Please install required dependencies.")
            ss.vlm_provider = None
            ss.vlm_model = None
        else:
            # Provider selection
            provider_options = list(model_providers.keys())
            selected_provider = st.selectbox("VLM Provider", provider_options, key="sidebar_vlm_provider")
            
            # Model selection
            available_models = list(model_providers[selected_provider])
            selected_vlm_model = st.selectbox("VLM Model", available_models, key="sidebar_vlm_model")
            
            # API key check and store in session state
            if selected_provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY") 
                if not api_key:
                    st.error("Please set OPENAI_API_KEY environment variable")
                    ss.vlm_provider = None
                    ss.vlm_model = None
                else:
                    st.success("OpenAI API key found")
                    ss.vlm_provider = "openai"
                    ss.vlm_model = selected_vlm_model
            elif selected_provider == "google":
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    st.error("Please set GOOGLE_API_KEY environment variable")
                    ss.vlm_provider = None
                    ss.vlm_model = None
                else:
                    st.success("Google API key found")
                    ss.vlm_provider = "google"
                    ss.vlm_model = selected_vlm_model
            else:
                # For other providers, set the values and let the test determine availability
                ss.vlm_provider = selected_provider
                ss.vlm_model = selected_vlm_model
            
            # Test model availability
            if ss.vlm_provider and ss.vlm_model:
                if st.button("🧪 Test VLM Model", key="sidebar_test_vlm"):
                    with st.spinner("Testing VLM model availability..."):
                        is_available, message = test_vlm_model(ss.vlm_provider, ss.vlm_model)
                        if is_available:
                            st.success(f"✅ {ss.vlm_provider}/{ss.vlm_model} available!")
                        else:
                            st.error(f"❌ Failed: {message}")
    else:
        # Clear VLM configuration when disabled
        ss.vlm_provider = None
        ss.vlm_model = None

    # Separator line
    st.markdown("---")

    st.markdown("**Select Data Type**")
    st.write("Image")  # Image only

    # Separator line
    st.markdown("---")

    subtype = st.selectbox("Select subtype (placeholder)", IMAGE_SUBTYPES, index=0)

    # Separator line
    st.markdown("---")

    st.caption(f"Dataset root: `{DATASET_ROOT}` (expects `real/` and `fake-v2/`)")

# ---------- Main layout ----------
left, right = st.columns([5,7])

with left:
    st.subheader("Data")
    c1, c2 = st.columns([1, 1])
    if c1.button("🎲 Get an image", use_container_width=True):
        ss.candidate = sample_random_image()
        ss.pred = None
    
    if ss.candidate:
        candidate_img = load_image(ss.candidate, max_w=1200)  # Make it larger
        if candidate_img:
            # Show prediction overlay if available
            if ss.pred and ss.candidate == ss.selected:
                interp = ss.pred["interpreted"]
                if "predicted_class" in interp and interp["predicted_class"] != "error":
                    overlay_text = f"{interp['predicted_class'].upper()} ({interp['confidence']:.2%})"
                    
                    # Color code based on correctness
                    if "real" in ss.selected.lower():
                        ground_truth = "real"
                    elif "fake" in ss.selected.lower():
                        ground_truth = "fake"
                    else:
                        ground_truth = "unknown"
                    
                    if ground_truth != "unknown" and interp["predicted_class"] == ground_truth:
                        overlay_text = f"✅ {overlay_text}"
                    elif ground_truth != "unknown":
                        overlay_text = f"❌ {overlay_text}"
                    
                    result_img = overlay_tag(candidate_img, overlay_text)
                    if result_img:
                        st.image(result_img, caption="Candidate with Prediction")
                    else:
                        st.image(candidate_img, caption="Candidate")
                else:
                    st.image(candidate_img, caption="Candidate")
            else:
                st.image(candidate_img, caption="Candidate")
            
            if c2.button("Use", use_container_width=True):
                ss.selected = ss.candidate
                ss.pred = None
        else:
            st.error("Failed to load candidate image")
    else:
        st.info("Click 'Get an image' to start")

with right:
    st.subheader("Prediction")
    if ss.selected:
        st.write(f"**Selected:** `{ss.selected}`")
        
        # Display ground truth from file path if available
        if "real" in ss.selected.lower():
            ground_truth = "real"
        elif "fake" in ss.selected.lower():
            ground_truth = "fake"
        else:
            ground_truth = "unknown"
        st.write(f"**Ground Truth:** {ground_truth}")
        
        if st.button("Run prediction", type="primary"):
            if not selected_model:
                st.error("Please select a model first")
            else:
                with st.spinner("Running inference…"):
                    t0 = time.time()
                    
                    # Use unified prediction function
                    result = unified_predict(selected_model, ss.selected, model_uri)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        dt = time.time() - t0
                        
                        # Store unified result format
                        ss.pred = {
                            "raw": json.dumps(result["raw"], indent=2, default=str) if not isinstance(result["raw"], str) else str(result["raw"]),
                            "interpreted": {
                                "predicted_class": result["predicted_class"],
                                "confidence": result["confidence"],
                                "probabilities": result["probabilities"],
                                "model_type": result["model_type"]
                            },
                            "latency": dt
                        }

        if ss.pred:
            # Remove image display from right column since it's already on the left
            # Just show the prediction metrics and details
            
            # Display prediction details
            if "interpreted" in ss.pred and ss.pred["interpreted"]["predicted_class"] != "error":
                interp = ss.pred["interpreted"]
                
                # Show prediction results prominently
                correct = interp["predicted_class"] == ground_truth
                info = f"**Prediction:**  {interp['predicted_class'].upper()}, **Ground Truth**: {ground_truth.upper()}" if ground_truth != "unknown" else interp['predicted_class'].upper()
                if correct:
                    st.success(info)
                else:
                    st.error(info)
                # col1, col2 = st.columns(2)
                # with col1:
                #     st.metric("Prediction", interp["predicted_class"].title(), 
                #              delta=None)#f"{interp['confidence']:.1%} confidence")
                # with col2:
                #     if ground_truth != "unknown":
                #         correct = interp["predicted_class"] == ground_truth
                #         # Use Streamlit's default styling with delta for visual feedback
                #         st.metric("Accuracy", "Correct" if correct else "Wrong",
                #                  delta=None)#"✓" if correct else "✗")
                
                # Probability breakdown
                if "probabilities" in interp:
                    st.write("**Probability Breakdown:**")
                    probs = interp["probabilities"]
                    for label, prob in probs.items():
                        st.progress(prob, text=f"{label.title()}: {prob:.1%}")
                
                # Show latency
                st.caption(f"Inference time: {ss.pred['latency']:.3f}s")
            
            # with st.expander("Raw prediction output"):
            #     st.code(ss.pred["raw"], language="json")
    else:
        st.info("Click **Get an image** → **Use** → **Run prediction**")

st.markdown("---")

with st.expander("Advanced Analysis"):
    st.write("**Note:** This dashboard expects MLflow models trained with the following specifications:")
    st.write("- Input: RGB images resized to 224×224 pixels, converted to tensors")
    st.write("- Output: Softmax probabilities [real_prob, fake_prob]") 
    st.write("- Labels: 0=real, 1=fake")
    st.write("")
    
    st.info("💡 **VLM Analysis Requirements:** To use VLM analysis, you need API keys for the selected provider. "
            "Copy `.env.example` to `.env` and add your API keys.")
    
    mode = st.radio("Analysis", ["VLM analysis", "Spectral comparison"], horizontal=True)
    if not ss.selected:
        st.info("Select an image first.")
    else:
        base = load_image(ss.selected, max_w=700)
        if base is None:
            st.error("Failed to load selected image")
        elif mode == "VLM analysis":
            # Debug information
            # st.write(f"**Debug:** VLM_AVAILABLE={VLM_AVAILABLE}, vlm_enabled={ss.vlm_enabled}, vlm_provider={ss.vlm_provider}, vlm_model={ss.vlm_model}")
            
            if not VLM_AVAILABLE:
                st.error("VLM functionality is not available. Please install required dependencies.")
            elif not ss.vlm_enabled:
                st.warning("Please enable VLM Analysis in the sidebar first.")
            elif not ss.vlm_provider or not ss.vlm_model:
                st.warning("Please configure VLM provider and model in the sidebar first.")
                st.info("1. Enable VLM Analysis in sidebar\n2. Select VLM Provider and Model\n3. Ensure API keys are configured")
            else:
                # Image display and analysis
                st.image(base, caption="Selected Image for VLM Analysis")
                
                # Show current VLM configuration
                st.info(f"**Current VLM Configuration:** {ss.vlm_provider}/{ss.vlm_model}")
                
                # Test model availability button
                # col1, col2 = st.columns([1, 1])
                # with col1:
                #     if st.button("🧪 Test VLM Model", key="main_test_vlm"):
                #         with st.spinner("Testing VLM model availability..."):
                #             is_available, message = test_vlm_model(ss.vlm_provider, ss.vlm_model)
                #             if is_available:
                #                 st.success(f"✅ {ss.vlm_provider}/{ss.vlm_model} available!")
                #             else:
                #                 st.error(f"❌ Failed: {message}")
                
                # with col2:
                if st.button("🔍 Run VLM Analysis", type="primary"):
                    with st.spinner("Running VLM analysis..."):
                        # Test model first
                        is_available, test_message = test_vlm_model(ss.vlm_provider, ss.vlm_model)
                        if not is_available:
                            st.error(f"Model not available: {test_message}")
                            st.info("Please check your API keys or test the model first.")
                        else:
                            analysis_result = vlm_analysis(base, subtype, ss.vlm_provider, ss.vlm_model)
                            st.markdown("### VLM Analysis Result:")
                            st.markdown(analysis_result)
        else:
            c1, c2 = st.columns(2)
            
            # Comparison selection controls at the top
            col_a, col_b = st.columns([1, 1])
            with col_a:
                comparison_type = st.radio("Pick comparison from:", 
                                         ["Any category", "Real images", "Fake images"], 
                                         horizontal=True, key="comparison_type")
            with col_b:
                if st.button("🎲 Pick comparison image"):
                    if comparison_type == "Real images":
                        cp = sample_random_image_from_category("real")
                    elif comparison_type == "Fake images":
                        cp = sample_random_image_from_category("fake")
                    else:
                        cp = sample_random_image()
                    
                    # Avoid same image
                    if cp == ss.selected:
                        if comparison_type == "Real images":
                            cp = sample_random_image_from_category("real")
                        elif comparison_type == "Fake images":
                            cp = sample_random_image_from_category("fake")
                        else:
                            cp = sample_random_image()
                    ss.compare = cp
            
            if ss.compare:
                comp = load_image(ss.compare, max_w=700)
                if comp is not None:
                    # Get ground truth for comparison image
                    if "real" in ss.compare.lower():
                        comp_ground_truth = "real"
                    elif "fake" in ss.compare.lower():
                        comp_ground_truth = "fake"
                    else:
                        comp_ground_truth = "unknown"
                    
                    # Original images row (aligned at top)
                    st.write("**Original Images:**")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(base, caption="Selected")
                    with c2:
                        st.image(comp, caption=f"Random from dataset (Ground Truth: {comp_ground_truth.title()})")
                    
                    # Spectrum images row (aligned at bottom)
                    st.write("**Frequency Spectrum Analysis:**")
                    col1_spectrum, col2_spectrum = st.columns(2)
                    
                    base_spectrum = spectrum_image(base)
                    comp_spectrum = spectrum_image(comp)
                    
                    with col1_spectrum:
                        if base_spectrum:
                            st.image(base_spectrum, caption="Selected • Spectrum")
                        else:
                            st.error("Failed to generate spectrum for selected image")
                    with col2_spectrum:
                        if comp_spectrum:
                            st.image(comp_spectrum, caption="Random • Spectrum")
                        else:
                            st.error("Failed to generate spectrum for comparison image")
                else:
                    st.error("Failed to load comparison image")
            else:
                st.info("Click the button above to pick a comparison image.")