#!/usr/bin/env python3
"""
ViT AI Image Detector 训练脚本

支持:
- 本地训练
- Google Colab (A100)
- MLflow 实验追踪
- 混合精度训练 (AMP)
- 早停机制
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# 检测环境
IN_COLAB = 'google.colab' in sys.modules

# 添加项目路径
if not IN_COLAB:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))

# MLflow (可选)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# WandB (可选)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# 本地导入
from src.models.vit_detector import ViTDetector
from src.dataset.combined_dataset import CombinedAIDataset, create_dataloaders
from src.training.config import TrainingConfig

# 评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Trainer:
    """ViT 训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print(f"🚀 ViT AI Image Detector Trainer")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"{'='*60}\n")
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 初始化模型
        self.model = ViTDetector(
            model_name=config.model_name,
            freeze_backbone=config.freeze_backbone,
            dropout=config.dropout,
        )
        
        # 初始化数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            limit=config.limit,
            include_artifact=config.include_artifact,
            include_flux=config.include_flux,
            balance_classes=config.balance_classes,
        )
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = self._create_scheduler(total_steps)
        
        # 混合精度
        self.scaler = GradScaler() if config.use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # 实验追踪初始化
        self._init_tracking()
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        config = self.config
        
        # 区分不同层的学习率
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        if config.optimizer == "adamw":
            return torch.optim.AdamW(
                optimizer_grouped_parameters, 
                lr=config.learning_rate,
            )
        elif config.optimizer == "adam":
            return torch.optim.Adam(
                optimizer_grouped_parameters, 
                lr=config.learning_rate,
            )
        else:
            return torch.optim.SGD(
                optimizer_grouped_parameters, 
                lr=config.learning_rate,
                momentum=0.9,
            )
    
    def _create_scheduler(self, total_steps: int):
        """创建学习率调度器"""
        config = self.config
        warmup_steps = int(total_steps * config.warmup_ratio)
        
        if config.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer, 
                T_max=total_steps - warmup_steps,
            )
        elif config.scheduler == "linear":
            from torch.optim.lr_scheduler import LinearLR
            return LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=total_steps,
            )
        else:
            return None
    
    def _init_tracking(self):
        """初始化实验追踪"""
        config = self.config
        
        if config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.set_experiment(config.experiment_name)
            mlflow.start_run(run_name=config.run_name)
            mlflow.log_params({
                "model_name": config.model_name,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "freeze_backbone": config.freeze_backbone,
            })
            print("MLflow tracking enabled")
        
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.experiment_name,
                name=config.run_name,
                config=config.__dict__,
            )
            print("WandB tracking enabled")
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics(metrics, step=step)
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=True,
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播 (混合精度)
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(images, labels=labels)
                    loss = outputs["loss"]
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            else:
                outputs = self.model(images, labels=labels)
                loss = outputs["loss"]
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            
            # 收集统计
            total_loss += loss.item()
            preds = outputs["logits"].argmax(-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            
            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy_score(all_labels, all_preds):.3f}",
            })
            
            self.global_step += 1
            
            # 定期记录
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_metrics({
                    "train/loss": loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                }, self.global_step)
            
            # 定期验证
            if self.global_step % self.config.eval_every_n_steps == 0:
                val_metrics = self.evaluate()
                self.model.train()
                
                self._log_metrics({
                    f"val/{k}": v for k, v in val_metrics.items()
                }, self.global_step)
        
        # Epoch 统计
        metrics = {
            "loss": total_loss / len(self.train_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="binary"),
            "recall": recall_score(all_labels, all_preds, average="binary"),
            "f1": f1_score(all_labels, all_preds, average="binary"),
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        if loader is None:
            loader = self.val_loader
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images, labels=labels)
            
            total_loss += outputs["loss"].item()
            preds = outputs["logits"].argmax(-1).cpu().tolist()
            probs = outputs["probs"][:, 1].cpu().tolist()  # P(fake)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs)
        
        metrics = {
            "loss": total_loss / len(loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, average="binary", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="binary", zero_division=0),
            "f1": f1_score(all_labels, all_preds, average="binary", zero_division=0),
        }
        
        # AUC (需要足够的正负样本)
        try:
            metrics["auc"] = roc_auc_score(all_labels, all_probs)
        except:
            metrics["auc"] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint_dir = self.config.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if is_best or not self.config.save_best_only:
            save_path = checkpoint_dir / f"epoch_{epoch+1}"
            self.model.save(str(save_path))
            
            # 保存训练状态
            state = {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_acc": self.best_val_acc,
                "metrics": metrics,
                "config": self.config.__dict__,
            }
            with open(checkpoint_dir / f"epoch_{epoch+1}_state.json", "w") as f:
                json.dump(state, f, indent=2, default=str)
        
        if is_best:
            best_dir = self.config.output_dir / "best_model"
            self.model.save(str(best_dir))
            print(f"  ✅ New best model saved! (acc={metrics['accuracy']:.4f})")
    
    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Starting training...")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")
        
        if self.config.dry_run:
            print("Dry run mode - skipping actual training")
            return
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.evaluate()
            
            # 记录
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            self._log_metrics({
                "epoch": epoch + 1,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_acc": train_metrics["accuracy"],
                "val/epoch_loss": val_metrics["loss"],
                "val/epoch_acc": val_metrics["accuracy"],
                "val/epoch_f1": val_metrics["f1"],
            }, self.global_step)
            
            # 保存检查点
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 最终测试
        print(f"\n{'='*60}")
        print("Final Evaluation on Test Set:")
        test_metrics = self.evaluate(self.test_loader)
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"{'='*60}")
        
        self._log_metrics({
            f"test/{k}": v for k, v in test_metrics.items()
        }, self.global_step)
        
        # 清理
        if self.config.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.end_run()
        
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        
        print(f"\n✅ Training complete! Best val accuracy: {self.best_val_acc:.4f}")
        print(f"   Model saved to: {self.config.output_dir / 'best_model'}")
        
        return test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train ViT AI Image Detector")
    
    # 数据参数
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing")
    
    # 模型参数
    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--freeze-backbone", action="store_true")
    
    # 训练参数
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--experiment-name", type=str, default="vit-ai-detector")
    
    # 调试参数
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--colab", action="store_true", help="Use Colab A100 optimized config")
    
    args = parser.parse_args()
    
    # 创建配置
    if args.colab or IN_COLAB:
        config = TrainingConfig.for_colab_a100()
    else:
        config = TrainingConfig()
    
    # 覆盖命令行参数
    if args.data_root:
        config.data_root = Path(args.data_root)
    if args.limit:
        config.limit = args.limit
    if args.model_name:
        config.model_name = args.model_name
    if args.freeze_backbone:
        config.freeze_backbone = True
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.no_amp:
        config.use_amp = False
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.dry_run:
        config.dry_run = True
    
    # 开始训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
