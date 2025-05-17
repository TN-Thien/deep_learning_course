import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import wandb
import statistics
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CIFAR-10 from local
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

# 2. Thuần CNN Model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

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
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    model.to(DEVICE)
    best_model_state = None
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
            best_model_state = model.state_dict()

    wandb.finish()
    return best_model_state

# 5. Evaluate on test set
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {correct / total:.4f}")

# 6. Main
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
        {"lr": 0.001, "batch_size": 128, "epochs": 40},
        {"lr": 0.002, "batch_size": 128, "epochs": 50},
        {"lr": 0.0005, "batch_size": 128, "epochs": 40},
    ]

    best_overall_state = None
    best_overall_acc = 0.0
    best_model_name = ""

    for idx, config in enumerate(configs):
        print(f"\nRunning Config {idx+1}: {config}")
        test_accuracies = []
        best_run_state = None
        best_run_acc = 0.0

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        for run in range(3):
            print(f"\nRun {run+1}")
            model = SimpleCNN()
            model_name = f"cnn_cfg{idx+1}_run{run+1}"

            model_state = train_model(model, train_loader, val_loader, config, model_name)
            model.load_state_dict(model_state)
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
            print(f"Test Accuracy: {acc:.4f}")
            test_accuracies.append(acc)

            if acc > best_run_acc:
                best_run_acc = acc
                best_run_state = model_state

        if best_run_acc > best_overall_acc:
            best_overall_acc = best_run_acc
            best_overall_state = best_run_state
            best_model_name = f"cnn_cfg{idx+1}"

        avg_acc = statistics.mean(test_accuracies)
        std_acc = statistics.stdev(test_accuracies)
        print(f"\nConfig {idx+1} - Average Test Accuracy: {avg_acc:.4f}, Std: {std_acc:.4f}")

    # Save best model across all configs
    torch.save(best_overall_state, "Week5/best_model.pth")
    print(f"\nBest overall model: {best_model_name} with Test Accuracy: {best_overall_acc:.4f}")
