#!/usr/bin/env python3
"""
Benchmark Script
对比评估 ViT 模型与 HuggingFace 开源模型
"""

import os
import sys
import argparse
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    pass

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_detector import ViTDetector
from src.models.hf_models import (
    AIOrNotHfModel, 
    SDXLDetectorHfModel, 
    AIVSHumanImageDetectorHfModel, 
    DafilabAIImageDetectorHfModel
)
from src.dataset.combined_dataset import IMG_EXTENSIONS

def get_image_files(directory: Path, limit: int = None):
    files = []
    if not directory.exists():
        return files
    
    for ext in IMG_EXTENSIONS:
        files.extend(list(directory.rglob(f"*{ext}")))
        files.extend(list(directory.rglob(f"*{ext.upper()}")))
    
    if limit:
        import random
        random.shuffle(files)
        files = files[:limit]
    return files

def load_models(device):
    models = {}
    
    # 1. Custom ViT
    # Try different potential locations for best_model
    potential_paths = [
        project_root / "outputs/best_model",
        Path("outputs/best_model"),
        Path("/content/outputs/best_model")
    ]
    
    vit_loaded = False
    for p in potential_paths:
        if p.exists():
            try:
                print(f"Loading custom ViT model from {p}...")
                vit = ViTDetector.load(str(p))
                vit.model.to(device)
                models["Custom ViT"] = vit
                vit_loaded = True
                break
            except Exception as e:
                print(f"Failed to load ViT from {p}: {e}")
    
    if not vit_loaded:
        print("⚠️ Custom ViT model not found. Run training first.")

    # 2. Open Source Models
    print("Loading Open Source Models (this may take time)...")
    try:
        models["Ateeqq/AI-vs-Human"] = AIVSHumanImageDetectorHfModel()
    except Exception as e: print(f"Skipped Ateeqq: {e}")
        
    try:
        models["Organika/SDXL"] = SDXLDetectorHfModel()
    except Exception as e: print(f"Skipped Organika: {e}")

    try:
        models["Nahrawy/AIorNot"] = AIOrNotHfModel()
    except Exception as e: print(f"Skipped Nahrawy: {e}")
    
    return models

def evaluate_model(model, real_imgs, fake_imgs_dict):
    """
    Evaluate a single model on multiple datasets
    fake_imgs_dict: {"Flux": [paths], "Midjourney": [paths], ...}
    """
    results = {}
    
    # 1. Test on Real
    preds = []
    
    # Batch processing could be optimized, but using single loop for compatibility
    for p in tqdm(real_imgs, desc="Eval Real", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            # Unified interface: predict() returns HfModelOutput(label, probs)
            res = model.predict(img) 
            
            # Label mapping: 0=real, 1=fake
            is_fake = 1 if res.label.lower() in ["fake", "ai", "artificial"] else 0
            # Warning: predict() logic in hf_models wrapper returns label string.
            # We assume label is correct.
            
            preds.append(is_fake)
        except Exception:
            continue
            
    # Accuracy on Real = (1 - fake_rate)
    acc_real = 1.0 - (sum(preds) / len(preds)) if preds else 0
    results["Real (Acc)"] = acc_real
    
    # 2. Test on Fakes (per category)
    for name, files in fake_imgs_dict.items():
        if not files:
            continue
            
        preds = []
        for p in tqdm(files, desc=f"Eval {name}", leave=False):
            try:
                img = Image.open(p).convert("RGB")
                res = model.predict(img)
                is_fake = 1 if res.label.lower() in ["fake", "ai", "artificial"] else 0
                preds.append(is_fake)
            except Exception:
                continue
        
        # Accuracy on Fake = fake_rate
        acc = sum(preds) / len(preds) if preds else 0
        results[f"{name} (Acc)"] = acc
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--limit", type=int, default=100, help="Images per category")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    
    # 1. Prepare Data
    print(f"Scanning data in {data_root}...")
    real_imgs = get_image_files(data_root / "real", args.limit)
    
    # Auto-detect fake subdirectories
    fake_imgs = {}
    
    # Flux
    if (data_root / "flux").exists():
        fake_imgs["Flux"] = get_image_files(data_root / "flux", args.limit)
    
    # Artifact subfolders if downloaded
    artifact_fake = data_root / "artifact/fake"
    if artifact_fake.exists():
         fake_imgs["ArtiFact"] = get_image_files(artifact_fake, args.limit)
            
    # Check if we have data
    if not real_imgs and not fake_imgs:
        print("❌ No data found. Please run download_datasets.py")
        return
        
    print(f"Found {len(real_imgs)} Real images")
    for k, v in fake_imgs.items():
        print(f"Found {len(v)} {k} images")
        
    # 2. Load Models
    models = load_models(args.device)
    if not models:
        print("❌ No models loaded.")
        return

    # 3. Evaluate
    all_results = []
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        try:
            metrics = evaluate_model(model, real_imgs, fake_imgs)
            metrics["Model"] = name
            all_results.append(metrics)
            print(f"  {metrics}")
        except Exception as e:
            print(f"  Failed: {e}")
            import traceback
            traceback.print_exc()

    # 4. Save & Plot
    if all_results:
        df = pd.DataFrame(all_results)
        # Reorder columns
        cols = ["Model"] + [c for c in df.columns if c != "Model"]
        df = df[cols]
        
        print("\nBenchmark Results:")
        try:
            print(df.to_markdown(index=False))
        except:
            print(df)
        
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
        
        # Plot
        try:
            plt.figure(figsize=(10, 6))
            # Melt for sns
            plot_df = df.melt(id_vars="Model", var_name="Dataset", value_name="Accuracy")
            sns.barplot(data=plot_df, x="Dataset", y="Accuracy", hue="Model")
            plt.title("AI Detection Benchmark: Custom ViT vs Open Source")
            plt.ylim(0, 1.0)
            plt.tight_layout()
            plt.savefig("benchmark_plot.png")
            print("Saved plot to benchmark_plot.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    main()
