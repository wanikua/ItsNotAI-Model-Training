from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from tqdm import tqdm

import mlflow
import mlflow.pytorch
import mlflow.models.signature
from mlflow.models.signature import infer_signature

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.dataset.test_dataset import AIRecognitionDataset
from src.models.model_api import OpenSourceModel, HuggingFaceModel, APIProvider

# mlflow.autolog()


class ResNetWithSoftmax(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.base(x)
        return self.softmax(x)


def train_model(model, train_loader, val_loader):

    with mlflow.start_run():
        # training loop
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 1
        log_every = 100
        correct = 0
        total = 0
        
        mlflow.log_param("model_name", "resnet18")
        mlflow.log_param("num_epochs", num_epochs)

        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in tqdm(enumerate(train_loader)):
                optimizer.zero_grad()
                outputs = model(images).squeeze()
                # print(outputs, labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

                if i % log_every == 0:          
                    acc = accuracy_score(labels.numpy(), preds.numpy())
                    mlflow.log_metric("train_loss", loss.item(), step=epoch * len(train_loader) + i)
                    mlflow.log_metric("train_accuracy", acc)

                    
            # validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
        
            with torch.inference_mode():
                val_preds = []
                val_labels = []
                for images, labels in tqdm(val_loader):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.numpy())
                    val_labels.extend(labels.numpy())

            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average="weighted")
            val_recall = recall_score(val_labels, val_preds, average="weighted")
            val_f1 = f1_score(val_labels, val_preds, average="weighted")

            mlflow.log_metric("val_loss", val_loss)
            mlflow.log_metric("val_accuracy", val_acc)
            mlflow.log_metric("val_precision", val_precision)
            mlflow.log_metric("val_recall", val_recall)
            mlflow.log_metric("val_f1", val_f1)

            print(f"[Epoch {epoch+1}] Train Loss: {loss.item():.4f}, Train Accuracy: {correct/total:.4f}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # log trained model
        sample_batch, _ = next(iter(val_loader))
        sample_output = model(sample_batch)
        signature = infer_signature(sample_batch.numpy(), sample_output.detach().numpy())
        model_info = mlflow.pytorch.log_model(
            model,
            artifact_path="test_model_resnet18",             # artifact path inside the run
            input_example=sample_batch[0].unsqueeze(0).numpy(),
            signature=signature,
            registered_model_name="test_model_resnet18"      # creates/updates the Registry entry
        )
        mlflow.set_logged_model_tags(
            model_info.model_id, {"Training Info": "Dummy model for testing purposes only"}
        )
        

def test_log_sample(model, loader, batch_size=4):
    """
        Randomly select a batch of images for validation and log the predictions
        as an image strip with predicted labels and ground truth
    """
    model.eval()
    images, labels = next(iter(loader))
    preds = model(images).argmax(dim=1)
    inv_map = {0: "real", 1: "fake"}
    
    fig, axes = plt.subplots(1, batch_size, figsize=(12, 6))
    for i in range(batch_size):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(f"P: {inv_map[preds[i].item()]}\nT: {inv_map[labels[i].item()]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.close()
    mlflow.log_artifact("sample_predictions.png")
    
        
def main():
    # test model
    model = ResNetWithSoftmax() #torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    

    train_dataset = AIRecognitionDataset(transform=transform, split="train", subset_limit=512)
    val_dataset = AIRecognitionDataset(transform=transform, split="val", subset_limit=128)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    mlflow.set_experiment("AI Image Detector v0")
    train_model(model, train_loader, val_loader)
    test_log_sample(model, val_loader, batch_size=4)
    client = mlflow.MlflowClient()
    out = []
    for rm in client.search_registered_models():
        out.append((rm.name, rm.latest_versions))
    print(out)
    
if __name__ == "__main__":
    main()