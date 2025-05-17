import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import wandb
import statistics
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CIFAR-10
class CIFAR10Custom(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        if train:
            files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            files = ["test_batch"]
        for file in files:
            with open(os.path.join(data_dir, file), 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.data.append(entry['data'])
                self.labels += entry['labels']
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32).astype(np.uint8)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform:
            img = self.transform(img)
        return img, label

# 2. Load ResNet18 với thay đổi đầu ra phù hợp
class ResNetForCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 3. Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (1 - self.smoothing) * nll + self.smoothing * smooth_loss

# 4. Train function
def train_model(model, train_loader, val_loader, config, model_name):
    wandb.init(project="cifar10-lab05", config=config, name=model_name)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    model.to(DEVICE)
    best_acc = 0.0
    for epoch in range(config["epochs"]):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.mean().item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "val_loss": val_loss / len(val_loader),
            "val_acc": val_acc
        }, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"Week5/best_model_{model_name}.pth")

    wandb.finish()
    return model

# 5. Main
if __name__ == "__main__":
    torch.manual_seed(42)
    data_dir = "Week5/cifar-10-batches-py"

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    full_dataset = CIFAR10Custom(data_dir, train=True, transform=train_transform)
    test_dataset = CIFAR10Custom(data_dir, train=False, transform=test_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    configs = [
        {"lr": 0.001, "batch_size": 128, "epochs": 30},
        {"lr": 0.0005, "batch_size": 128, "epochs": 40},
        {"lr": 0.0003, "batch_size": 128, "epochs": 50},
    ]

    for idx, config in enumerate(configs):
        print(f"\nRunning Config {idx+1}: {config}")
        accs = []

        for run in range(3):
            print(f"Run {run+1} for Config {idx+1}")
            model_name = f"resnet18_cfg{idx+1}_run{run+1}"

            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
            test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

            model = ResNetForCIFAR10()
            train_model(model, train_loader, val_loader, config, model_name)

            model.load_state_dict(torch.load(f"Week5/best_model_{model_name}.pth"))
            model.to(DEVICE)
            model.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total
            accs.append(acc)
            print(f"Test Accuracy: {acc:.4f}")

        avg = statistics.mean(accs)
        std = statistics.stdev(accs)
        print(f"\nConfig {idx+1} - Avg Acc: {avg:.4f}, Std: {std:.4f}")
