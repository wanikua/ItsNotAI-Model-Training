"""
ViT (Vision Transformer) AI Image Detector
基于 google/vit-base-patch16-224 微调
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from transformers import ViTForImageClassification, ViTImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Install with: pip install transformers")

# Local imports with fallback
try:
    from src.models.model_api import HuggingFaceModel, HfModelOutput
    from src.utils.model_utils import sanitize_label, get_device
except ImportError:
    # Fallback for Colab or standalone usage
    @dataclass
    class HfModelOutput:
        label: str
        probs: list
    
    class HuggingFaceModel:
        pass
    
    def sanitize_label(labels):
        mapping = {"real": "real", "fake": "fake", "ai": "fake", "human": "real"}
        return [mapping.get(l.lower(), l.lower()) for l in labels]
    
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ViTDetector(nn.Module):
    """
    Vision Transformer for AI Image Detection
    
    基于预训练 ViT 微调的二分类模型
    输出: real (0) / fake (1)
    """
    
    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = get_device()
        
        # 加载预训练模型
        print(f"Loading ViT model: {model_name}")
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,  # 因为我们改变了分类头
        )
        
        # 加载图像处理器
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # 可选: 冻结 backbone
        if freeze_backbone:
            for param in self.model.vit.parameters():
                param.requires_grad = False
            print("Backbone frozen, only training classifier head")
        
        # 添加 dropout
        if dropout > 0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.model.config.hidden_size, num_labels),
            )
        
        self.model.to(self.device)
        self.labels = ["real", "fake"]
        
        print(f"ViTDetector initialized on {self.device}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: (B, C, H, W) 图像张量
            labels: (B,) 可选标签用于计算损失
        
        Returns:
            字典包含 logits, (loss if labels provided), probs
        """
        outputs = self.model(
            pixel_values=pixel_values.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        
        result = {
            "logits": outputs.logits,
            "probs": torch.softmax(outputs.logits, dim=-1),
        }
        
        if labels is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def predict(
        self,
        image: Image.Image,
        return_probs: bool = True,
    ) -> HfModelOutput:
        """
        单图预测 (兼容现有 HuggingFaceModel 接口)
        
        Args:
            image: PIL Image
            return_probs: 是否返回概率
        
        Returns:
            HfModelOutput with label and probs
        """
        self.eval()
        
        # 预处理
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            probs = outputs["probs"][0].cpu().tolist()
            pred_idx = outputs["logits"].argmax(-1).item()
            label = self.labels[pred_idx]
        
        return HfModelOutput(
            label=sanitize_label([label])[0],
            probs=probs,
        )
    
    def predict_batch(
        self,
        images: list,
    ) -> list:
        """
        批量预测
        
        Args:
            images: PIL Image 列表
        
        Returns:
            HfModelOutput 列表
        """
        self.eval()
        
        # 批量预处理
        inputs = self.processor(images, return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(pixel_values)
            probs_batch = outputs["probs"].cpu().tolist()
            preds = outputs["logits"].argmax(-1).cpu().tolist()
        
        results = []
        for pred_idx, probs in zip(preds, probs_batch):
            label = self.labels[pred_idx]
            results.append(HfModelOutput(
                label=sanitize_label([label])[0],
                probs=probs,
            ))
        
        return results
    
    def save(self, save_path: str):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, model_path: str, **kwargs) -> "ViTDetector":
        """加载保存的模型"""
        instance = cls.__new__(cls)
        super(ViTDetector, instance).__init__()
        
        instance.device = get_device()
        instance.model = ViTForImageClassification.from_pretrained(model_path)
        instance.processor = ViTImageProcessor.from_pretrained(model_path)
        instance.model.to(instance.device)
        instance.labels = ["real", "fake"]
        instance.num_labels = 2
        
        print(f"Model loaded from {model_path}")
        return instance


class ViTDetectorHfModel(HuggingFaceModel):
    """
    ViT Detector 适配器
    用于与现有 Dashboard 和评估框架集成
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        if model_path:
            self.detector = ViTDetector.load(model_path)
        else:
            self.detector = ViTDetector()
        
        print(f"[{self.__class__.__name__}] Model initialization done.")
    
    def predict(self, img: Image.Image, *, with_probs: bool = True) -> HfModelOutput:
        return self.detector.predict(img, return_probs=with_probs)


# 快捷函数
def create_vit_detector(
    pretrained: bool = True,
    freeze_backbone: bool = False,
    **kwargs,
) -> ViTDetector:
    """创建 ViT Detector 实例"""
    return ViTDetector(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **kwargs,
    )


if __name__ == "__main__":
    # 测试模型
    print("Testing ViTDetector...")
    
    model = ViTDetector()
    
    # 创建测试图像
    test_img = Image.new('RGB', (224, 224), color='red')
    
    # 单图预测
    result = model.predict(test_img)
    print(f"Single prediction: {result}")
    
    # 批量预测
    batch_results = model.predict_batch([test_img, test_img])
    print(f"Batch prediction: {batch_results}")
    
    print("\n✅ ViTDetector test passed!")
