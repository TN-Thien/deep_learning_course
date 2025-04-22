import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đọc dữ liệu
df = pd.read_csv("Week3/data/california_housing.csv")

# Tách features và target
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tách tập train và test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

# Tạo Dataset & DataLoader
class HousingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = HousingDataset(X_train, y_train)
test_dataset = HousingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# MLP model
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Hàm đánh giá mô hình với MAE, RMSE và R²
def evaluate_model(model, test_loader, loss_fn):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()

            all_preds.append(y_pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    y_pred_all = np.vstack(all_preds)
    y_true_all = np.vstack(all_targets)

    # Tính MAE, RMSE và R²
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)

    avg_loss = total_loss / len(test_loader)
    return avg_loss, mae, rmse, r2

# Huấn luyện tích hợp Wandb
def train_with_wandb(config=None):
    with wandb.init(config=config):
        config = wandb.config
        model = MLP(input_size=X_train.shape[1]).to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            wandb.log({"epoch": epoch + 1, "train_loss": avg_loss})
        
        # Đánh giá trên tập test
        test_loss, mae, rmse, r2 = evaluate_model(model, test_loader, loss_fn)
        wandb.log({"test_loss": test_loss, "test_mae": mae, "test_rmse": rmse, "test_r2": r2})
        
        return test_loss, mae, rmse, r2

# Các cấu hình siêu tham số
configs = [
    {"lr": 1e-3, "epochs": 200},
    {"lr": 5e-4, "epochs": 200},
    {"lr": 1e-4, "epochs": 250},
    {"lr": 1e-3, "epochs": 450},
    {"lr": 1e-2, "epochs": 300}
]

# Chạy thử 5 lần với mỗi cấu hình
results = []

for config in configs:
    scores = []
    for run_id in range(5):
        print(f"Running config: {config} - Run {run_id + 1}/5")
        test_loss, mae, rmse, r2 = train_with_wandb(config)
        scores.append((mae, rmse, r2))
    
    mean_scores = np.mean(scores, axis=0)
    std_scores = np.std(scores, axis=0)
    print(f"Config {config} - Mean MAE: {mean_scores[0]:.4f}, Mean RMSE: {mean_scores[1]:.4f}, Mean R²: {mean_scores[2]:.4f}")
    print(f"Config {config} - Std MAE: {std_scores[0]:.4f}, Std RMSE: {std_scores[1]:.4f}, Std R²: {std_scores[2]:.4f}")
    results.append((config, mean_scores, std_scores))
