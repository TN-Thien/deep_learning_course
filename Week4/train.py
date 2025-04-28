import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import random

# Cố định seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset
class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx].toarray(), dtype=torch.float32).squeeze(0), torch.tensor(self.labels[idx])

# Model
class NewsClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NewsClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# Load dữ liệu
df_vnexpress = pd.read_csv("Week4/data/Data_vnexpress_processed.csv")
df_dantri = pd.read_csv("Week4/data/Data_dantri_processed.csv")

# Gộp dữ liệu
df_all = pd.concat([df_vnexpress, df_dantri], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Load label mapping
with open("Week4/data/label_mapping.json", encoding="utf-8") as f:
    label_mapping = json.load(f)
    id2label = {int(k): v for k, v in label_mapping["id2label"].items()}

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform((df_all['title'] + " " + df_all['content']).values)
y = df_all['label'].values

# Chia cố định tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Tạo dataset/dataloader cố định
train_dataset = NewsDataset(X_train, y_train)
test_dataset = NewsDataset(X_test, y_test)

# Các config để thử
configs = [
    {"lr": 0.01, "batch_size": 32, "hidden_size": 128},
    {"lr": 0.005, "batch_size": 64, "hidden_size": 256},
    {"lr": 0.001, "batch_size": 32, "hidden_size": 64}
]

results = []

for i, config in enumerate(configs):
    accuracies = []

    for run in range(3):
        wandb.init(project="vnexpress-news-classification", config=config, name=f"Config{i+1}_Run{run+1}")

        # Tạo DataLoader mỗi run
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

        # Model
        model = NewsClassifier(input_size=5000, hidden_size=config["hidden_size"], num_classes=len(id2label)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        best_acc = 0.0

        for epoch in range(10):
            model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            wandb.log({"epoch": epoch, "loss": total_loss / len(train_loader)})

            # Validation
            model.eval()
            preds = []
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    predicted = torch.argmax(outputs, dim=1)
                    preds.extend(predicted.cpu().numpy())

            acc = accuracy_score(y_test, preds)
            wandb.log({"val_accuracy": acc})

            if acc > best_acc:
                best_acc = acc

        accuracies.append(best_acc)
        wandb.finish()

    avg_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    results.append((config, avg_acc, std_acc))

# Tổng kết
print("\nTổng kết:")
for i, (config, avg, std) in enumerate(results):
    print(f"Config {i+1}: {config}")
    print(f"  Accuracy trung bình: {avg:.4f}, Độ lệch chuẩn: {std:.4f}")
