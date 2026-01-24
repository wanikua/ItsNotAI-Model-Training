"""
ViT (Vision Transformer) AI Image Detector
基于 google/vit-base-patch16-224 微调

支持两种模式:
1. 二分类: real (0) / fake (1)
2. 多分类: 识别具体的生成器/来源 (StyleGAN2, Stable Diffusion, etc.)
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field

try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
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


@dataclass
class MultiClassOutput:
    """多分类预测输出"""
    predicted_source: str           # 最可能的来源/生成器
    is_real: bool                   # 是否为真实图片
    confidence: float               # 置信度
    all_probs: Dict[str, float]     # 所有来源的概率
    top_k: List[Tuple[str, float]] = field(default_factory=list)  # Top-K 预测


class ViTDetector(nn.Module):
    """
    Universal Vision Transformer Detector (ViT, BEiT, Swin, etc.)

    基于 AutoModel 支持多种 Transformer 架构

    模式:
    - 二分类 (num_labels=2): real (0) / fake (1)
    - 多分类 (num_labels>2): 识别具体生成器/来源
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_labels: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        source_names: Optional[List[str]] = None,  # 多分类时的来源名称列表
        source_is_real: Optional[Dict[str, bool]] = None,  # 每个来源是否为真实
    ):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install with: pip install transformers")

        self.model_name = model_name
        self.num_labels = num_labels
        self.device = get_device()
        self.multiclass = num_labels > 2

        # 来源信息 (多分类模式)
        if source_names:
            self.source_names = source_names
            self.source_is_real = source_is_real or {}
        else:
            self.source_names = ["real", "fake"] if num_labels == 2 else [f"class_{i}" for i in range(num_labels)]
            self.source_is_real = {"real": True} if num_labels == 2 else {}

        # 加载预训练模型
        print(f"Loading Model: {model_name}")

        # 针对 BEiT 等模型的高级配置
        config_kwargs = {}
        if drop_path_rate > 0:
            config_kwargs["drop_path_rate"] = drop_path_rate

        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            **config_kwargs
        )

        # 加载图像处理器
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # 可选: 冻结 backbone
        if freeze_backbone:
            backbone = None
            for name, module in self.model.named_children():
                if name not in ["classifier", "head", "fc"]:
                    backbone = module
                    break

            if backbone:
                for param in backbone.parameters():
                    param.requires_grad = False
                print(f"Backbone ({type(backbone).__name__}) frozen")
            else:
                print("Warning: Could not identify backbone to freeze")

        # 添加 dropout
        if dropout > 0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.model.config.hidden_size, num_labels),
            )

        self.model.to(self.device)
        self.labels = self.source_names  # 兼容旧接口

        print(f"ViTDetector initialized on {self.device}")
        print(f"  Mode: {'Multi-class (' + str(num_labels) + ' sources)' if self.multiclass else 'Binary'}")
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

        # 二分类模式: 返回 real/fake
        if not self.multiclass:
            return HfModelOutput(
                label=sanitize_label([label])[0],
                probs=probs,
            )

        # 多分类模式: 也返回 HfModelOutput 但 label 是来源名称
        return HfModelOutput(
            label=label,
            probs=probs,
        )

    def predict_multiclass(
        self,
        image: Image.Image,
        top_k: int = 5,
    ) -> MultiClassOutput:
        """
        多分类预测 - 返回每个生成器的概率

        Args:
            image: PIL Image
            top_k: 返回 Top-K 预测

        Returns:
            MultiClassOutput 包含:
            - predicted_source: 最可能的来源
            - is_real: 是否为真实图片
            - confidence: 置信度
            - all_probs: 所有来源的概率字典
            - top_k: Top-K 预测列表
        """
        self.eval()

        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            outputs = self.forward(pixel_values)
            probs = outputs["probs"][0].cpu().tolist()
            pred_idx = outputs["logits"].argmax(-1).item()

        # 构建概率字典
        all_probs = {name: prob for name, prob in zip(self.source_names, probs)}

        # Top-K 预测
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_results = sorted_probs[:top_k]

        # 预测的来源
        predicted_source = self.source_names[pred_idx]
        is_real = self.source_is_real.get(predicted_source, False)

        return MultiClassOutput(
            predicted_source=predicted_source,
            is_real=is_real,
            confidence=probs[pred_idx],
            all_probs=all_probs,
            top_k=top_k_results,
        )

    def get_real_vs_fake_prob(
        self,
        image: Image.Image,
    ) -> Tuple[float, float]:
        """
        聚合计算 real vs fake 总概率

        Returns:
            (real_prob, fake_prob) 元组
        """
        result = self.predict_multiclass(image)

        real_prob = sum(
            prob for name, prob in result.all_probs.items()
            if self.source_is_real.get(name, False)
        )
        fake_prob = 1.0 - real_prob

        return real_prob, fake_prob
    
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
        """保存模型和来源信息"""
        import json
        import os

        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        # 保存来源信息
        meta = {
            "source_names": self.source_names,
            "source_is_real": self.source_is_real,
            "num_labels": self.num_labels,
            "multiclass": self.multiclass,
        }
        with open(os.path.join(save_path, "source_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Model saved to {save_path}")

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "ViTDetector":
        """加载保存的模型"""
        import json
        import os

        instance = cls.__new__(cls)
        super(ViTDetector, instance).__init__()

        instance.device = get_device()
        instance.model = AutoModelForImageClassification.from_pretrained(model_path)
        instance.processor = AutoImageProcessor.from_pretrained(model_path)
        instance.model.to(instance.device)

        # 加载来源信息
        meta_path = os.path.join(model_path, "source_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            instance.source_names = meta.get("source_names", ["real", "fake"])
            instance.source_is_real = meta.get("source_is_real", {"real": True})
            instance.num_labels = meta.get("num_labels", 2)
            instance.multiclass = meta.get("multiclass", False)
        else:
            # 兼容旧模型
            instance.source_names = ["real", "fake"]
            instance.source_is_real = {"real": True}
            instance.num_labels = 2
            instance.multiclass = False

        instance.labels = instance.source_names

        print(f"Model loaded from {model_path}")
        print(f"  Mode: {'Multi-class (' + str(instance.num_labels) + ' sources)' if instance.multiclass else 'Binary'}")
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
