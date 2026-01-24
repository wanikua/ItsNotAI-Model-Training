# ViT AI Image Detector - Colab Training Notebook

这个 notebook 用于在 Google Colab (A100) 上训练 AI 图像检测模型。

## 🚀 快速开始

**预计训练时间 (A100):**

- 50K 图片，5 epochs: ~30-45 分钟
- 100K 图片，5 epochs: ~1-1.5 小时

---

## 1. 环境设置

```python
# 检查 GPU
!nvidia-smi

# 安装依赖
!pip install -q torch torchvision transformers datasets huggingface_hub
!pip install -q mlflow scikit-learn tqdm pillow kagglehub

# 克隆项目
!git clone https://github.com/YOUR_USERNAME/ItsNotAi-model-backend.git
%cd ItsNotAi-model-backend

# 或者上传项目文件
# from google.colab import files
# uploaded = files.upload()
```

---

## 2. 下载数据集

```python
import os
import sys
sys.path.insert(0, '/content/ItsNotAi-model-backend')

from src.dataset.download_datasets import prepare_combined_dataset, verify_dataset

# 设置 Kaggle API (需要上传 kaggle.json)
# from google.colab import files
# files.upload()  # 上传 kaggle.json
# !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 下载数据集
DATA_ROOT = '/content/data'
prepare_combined_dataset(DATA_ROOT)
verify_dataset(DATA_ROOT)
```

### 手动上传数据 (如果自动下载失败)

```python
from google.colab import drive
drive.mount('/content/drive')

# 假设你的数据在 Google Drive
!cp -r "/content/drive/MyDrive/datasets/artifact" /content/data/
!cp -r "/content/drive/MyDrive/datasets/flux" /content/data/
```

---

## 3. 快速测试 (验证环境)

```python
# 测试数据集加载
from src.dataset.combined_dataset import CombinedAIDataset

ds = CombinedAIDataset(
    data_root='/content/data',
    split='train',
    limit=100,
)
print(f"Dataset size: {len(ds)}")

if len(ds) > 0:
    img, label = ds[0]
    print(f"Sample shape: {img.shape}, label: {label}")
```

```python
# 测试模型
from src.models.vit_detector import ViTDetector
from PIL import Image

model = ViTDetector()
test_img = Image.new('RGB', (224, 224), color='red')
result = model.predict(test_img)
print(f"Test prediction: {result}")
```

---

## 4. 开始训练

### 方式 A: 使用脚本

```python
!python src/training/train_vit.py \
    --colab \
    --data-root /content/data \
    --batch-size 128 \
    --epochs 5 \
    --output-dir /content/outputs \
    --experiment-name vit-ai-detector
```

### 方式 B: 使用 Python 代码 (更灵活)

```python
from src.training.config import TrainingConfig
from src.training.train_vit import Trainer

# A100 优化配置
config = TrainingConfig.for_colab_a100()

# 自定义参数
config.data_root = '/content/data'
config.output_dir = '/content/outputs'
config.num_epochs = 5
config.batch_size = 128  # A100 可以更大
config.learning_rate = 2e-5
config.use_mlflow = True
config.experiment_name = "vit-ai-detector-colab"

# 开始训练
trainer = Trainer(config)
test_metrics = trainer.train()

print("\n🎉 训练完成!")
print(f"测试集指标: {test_metrics}")
```

---

## 5. 保存模型到 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# 复制训练好的模型到 Drive
!cp -r /content/outputs/best_model "/content/drive/MyDrive/vit-ai-detector-model"

print("✅ 模型已保存到 Google Drive!")
```

---

## 6. 测试训练好的模型

```python
from src.models.vit_detector import ViTDetector
from PIL import Image
import requests
from io import BytesIO

# 加载模型
model = ViTDetector.load('/content/outputs/best_model')

# 测试网络图片
def test_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    result = model.predict(img)
    return result

# 测试一些图片
test_urls = [
    "https://example.com/real_photo.jpg",  # 替换为真实图片URL
    "https://example.com/ai_generated.jpg",  # 替换为AI生成图片URL
]

for url in test_urls:
    try:
        result = test_image(url)
        print(f"URL: {url[:50]}...")
        print(f"  Prediction: {result.label}")
        print(f"  Confidence: {max(result.probs):.2%}")
    except Exception as e:
        print(f"Error with {url}: {e}")
```

---

## 7. 上传模型到 HuggingFace Hub (可选)

```python
from huggingface_hub import HfApi, login

# 登录 HuggingFace
login()  # 需要 token

# 上传模型
api = HfApi()
api.upload_folder(
    folder_path="/content/outputs/best_model",
    repo_id="YOUR_USERNAME/vit-ai-image-detector",
    repo_type="model",
)

print("✅ 模型已上传到 HuggingFace!")
```

---

## 📊 训练监控

### 查看 MLflow 日志

```python
import mlflow

# 列出所有运行
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("vit-ai-detector-colab")

if experiment:
    runs = client.search_runs(experiment.experiment_id)
    for run in runs:
        print(f"Run: {run.info.run_id}")
        print(f"  Metrics: {run.data.metrics}")
```

### 可视化训练曲线

```python
import matplotlib.pyplot as plt

# 如果保存了训练历史
# history = trainer.history
# plt.plot(history['train_loss'], label='Train Loss')
# plt.plot(history['val_loss'], label='Val Loss')
# plt.legend()
# plt.show()
```

---

## ⚠️ 常见问题

### Q: CUDA out of memory

```python
# 减小 batch size
config.batch_size = 32  # 从 128 减小

# 或者启用梯度累积
config.gradient_accumulation_steps = 4
```

### Q: 数据集太小

```python
# 使用数据增强
from src.dataset.combined_dataset import get_train_transforms
# 已经默认启用了增强
```

### Q: 训练不收敛

```python
# 降低学习率
config.learning_rate = 1e-5

# 或增加 warmup
config.warmup_ratio = 0.2
```

---

**Happy Training! 🚀**
