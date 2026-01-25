"""
训练配置
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import sys


@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 模型配置
    # SOTA 推荐: microsoft/beit-large-patch16-224 (需要 timm)
    # 备选: google/vit-large-patch16-224
    model_name: str = "microsoft/beit-large-patch16-224"
    num_labels: int = 2
    freeze_backbone: bool = False
    dropout: float = 0.1
    drop_path_rate: float = 0.1  # Stochastic Depth for large models

    # 预训练模型 (从已有模型继续训练)
    pretrained_model_path: Optional[str] = None  # HF Hub ID 或本地路径, e.g. "boluobobo/ItsNotAI-ai-detector-v1"
    freeze_pretrained_backbone: bool = False  # 是否冻结预训练 backbone (只训练新的 binary head)

    # 数据配置
    data_root: Optional[Path] = None
    img_size: int = 224
    include_artifact: bool = True
    include_flux: bool = True
    balance_classes: bool = True
    multiclass: bool = False  # 多分类模式: 识别具体生成器
    dual_head: bool = False   # 双头模式: 多分类 + 二分类
    strong_augmentation: bool = True  # 强数据增强 (JPEG压缩, 噪声等)

    # 二分类 loss 权重 (提升真实照片识别率)
    # real_weight > 1.0 会让模型更倾向于预测为真实
    binary_class_weights: Optional[List[float]] = None  # [real_weight, ai_weight], e.g. [1.5, 1.0]

    # 双头 loss 权重平衡
    # 控制多分类和二分类 loss 的权重比例
    multiclass_loss_weight: float = 1.0  # 多分类 loss 权重
    binary_loss_weight: float = 1.0      # 二分类 loss 权重
    
    # 训练配置
    batch_size: int = 32  # Large model 需要更小 batch 或更多显存
    num_epochs: int = 10  # Large model 需要更多 epochs
    learning_rate: float = 1e-5  # Large model 需要更小 LR
    weight_decay: float = 0.05
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4  # 模拟大 Batch
    max_grad_norm: float = 1.0
    
    # 损失函数
    loss_type: str = "focal"  # ce, focal
    label_smoothing: float = 0.1
    
    # 优化器配置
    optimizer: str = "adamw"  # adamw, adam, sgd
    scheduler: str = "cosine"  # cosine, linear, constant
    
    # 训练控制
    early_stopping_patience: int = 3
    save_every_n_epochs: int = 1
    eval_every_n_steps: int = 500
    log_every_n_steps: int = 50
    
    # 混合精度
    use_amp: bool = True  # A100 效果很好
    
    # 数据加载
    num_workers: int = 4
    pin_memory: bool = True
    
    # 实验追踪
    use_mlflow: bool = True
    use_wandb: bool = False
    experiment_name: str = "vit-ai-detector"
    run_name: Optional[str] = None

    # Hugging Face Hub Upload
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    
    # 输出
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    save_best_only: bool = True
    
    # 调试
    limit: Optional[int] = None  # 限制样本数用于快速测试
    dry_run: bool = False  # 不实际训练
    seed: int = 42
    
    def __post_init__(self):
        """初始化后处理"""
        if self.data_root is None:
            if 'google.colab' in sys.modules:
                self.data_root = Path('/content/data')
            else:
                self.data_root = Path('data')
        
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def for_colab_a100(cls) -> "TrainingConfig":
        """Colab A100 优化配置"""
        return cls(
            batch_size=128,  # A100 40GB 可以更大
            use_amp=True,
            num_workers=2,  # Colab 限制
            data_root=Path('/content/data'),
            output_dir=Path('/content/outputs'),
        )
    
    @classmethod
    def for_quick_test(cls) -> "TrainingConfig":
        """快速测试配置"""
        return cls(
            batch_size=16,
            num_epochs=1,
            limit=200,
            eval_every_n_steps=50,
            log_every_n_steps=10,
        )

    @classmethod
    def for_multiclass(cls) -> "TrainingConfig":
        """多分类配置 - 识别具体生成器"""
        return cls(
            multiclass=True,
            balance_classes=False,  # 多分类不平衡类别
            num_epochs=15,  # 多分类需要更多 epochs
            learning_rate=5e-6,  # 更小的学习率
        )

    @classmethod
    def for_colab_multiclass(cls) -> "TrainingConfig":
        """Colab A100 多分类配置"""
        return cls(
            multiclass=True,
            balance_classes=False,
            batch_size=64,  # A100 可以大一些
            use_amp=True,
            num_workers=2,
            num_epochs=15,
            learning_rate=5e-6,
            data_root=Path('/content/data'),
            output_dir=Path('/content/outputs'),
        )

    @classmethod
    def for_dual_head(cls) -> "TrainingConfig":
        """双头模式配置 - 多分类 + 二分类"""
        return cls(
            multiclass=True,
            dual_head=True,
            balance_classes=False,
            num_epochs=10,
            learning_rate=5e-6,
            binary_class_weights=[1.5, 1.0],  # 提升真实照片识别率
        )

    @classmethod
    def for_colab_dual_head(cls) -> "TrainingConfig":
        """Colab A100 双头模式配置"""
        return cls(
            multiclass=True,
            dual_head=True,
            balance_classes=False,
            batch_size=64,
            use_amp=True,
            num_workers=2,
            num_epochs=10,
            learning_rate=5e-6,
            data_root=Path('/content/data'),
            output_dir=Path('/content/outputs'),
            binary_class_weights=[1.5, 1.0],  # 提升真实照片识别率
        )

    @classmethod
    def for_finetune_v1(cls) -> "TrainingConfig":
        """从 v1 模型微调 (添加双头)"""
        return cls(
            pretrained_model_path="boluobobo/ItsNotAI-ai-detector-v1",
            multiclass=True,
            dual_head=True,
            balance_classes=False,
            num_epochs=5,  # 微调不需要太多 epochs
            learning_rate=1e-5,  # 微调用更小的学习率
            binary_class_weights=[1.5, 1.0],
        )

    @classmethod
    def for_colab_finetune_v1(cls) -> "TrainingConfig":
        """Colab A100 从 v1 模型微调"""
        return cls(
            pretrained_model_path="boluobobo/ItsNotAI-ai-detector-v1",
            multiclass=True,
            dual_head=True,
            balance_classes=False,
            batch_size=64,
            use_amp=True,
            num_workers=2,
            num_epochs=5,
            learning_rate=1e-5,
            data_root=Path('/content/data'),
            output_dir=Path('/content/outputs'),
            binary_class_weights=[1.5, 1.0],
        )

    @classmethod
    def for_colab_a100_high_ram(cls) -> "TrainingConfig":
        """Colab A100 High RAM (80GB) 优化配置"""
        return cls(
            pretrained_model_path="boluobobo/ItsNotAI-ai-detector-v1",
            multiclass=True,
            dual_head=True,
            balance_classes=False,
            # A100 80GB 可以用更大 batch
            batch_size=256,
            gradient_accumulation_steps=1,  # 不需要梯度累积
            use_amp=True,
            num_workers=4,  # 更多 workers
            pin_memory=True,
            num_epochs=5,
            learning_rate=2e-5,  # 大 batch 可以用稍大学习率
            warmup_ratio=0.05,  # 大 batch 需要更少 warmup
            data_root=Path('/content/data'),
            output_dir=Path('/content/outputs'),
            binary_class_weights=[1.5, 1.0],
            strong_augmentation=True,
            eval_every_n_steps=200,  # 更频繁评估
        )


@dataclass
class EvalConfig:
    """评估配置"""
    
    model_path: str = "outputs/best_model"
    data_root: Optional[Path] = None
    batch_size: int = 64
    num_workers: int = 4
    
    # 评估设置
    split: str = "test"
    save_predictions: bool = True
    output_dir: Path = field(default_factory=lambda: Path("eval_results"))
