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
    model_name: str = "google/vit-base-patch16-224"
    num_labels: int = 2
    freeze_backbone: bool = False
    dropout: float = 0.1
    
    # 数据配置
    data_root: Optional[Path] = None
    img_size: int = 224
    include_artifact: bool = True
    include_flux: bool = True
    balance_classes: bool = True
    
    # 训练配置
    batch_size: int = 64  # A100 可以用更大的 batch
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
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
