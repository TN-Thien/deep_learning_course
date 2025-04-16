import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import numpy as np

# Thêm đường dẫn NLTK
# nltk.download('punkt')
# nltk.download('punkt_tab')
nltk.data.path.append("C:/Users/Acer/nltk_data")

# Đọc dữ liệu
df = pd.read_csv("Week2/IMDB Dataset.csv")
df_train, df_test = train_test_split(df, train_size=5000, test_size=5000, stratify=df['sentiment'], random_state=42)

# Tiền xử lý văn bản
def preprocess(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text.lower())
    return tokens

df_train['tokens'] = df_train['review'].apply(preprocess)
df_test['tokens'] = df_test['review'].apply(preprocess)

# Tạo vocab
all_tokens = [token for tokens in df_train['tokens'] for token in tokens]
vocab = ["<PAD>", "<UNK>"] + [word for word, freq in Counter(all_tokens).most_common(10000)]
word2idx = {word: idx for idx, word in enumerate(vocab)}

# Mã hóa tokens
def encode(tokens):
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokens]

df_train['input_ids'] = df_train['tokens'].apply(encode)
df_test['input_ids'] = df_test['tokens'].apply(encode)

# Mã hóa nhãn
label_encoder = LabelEncoder()
df_train['label'] = label_encoder.fit_transform(df_train['sentiment'])
df_test['label'] = label_encoder.transform(df_test['sentiment'])

# Dataset và DataLoader
class IMDBDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True)
    return padded_texts, torch.tensor(labels)

train_dataset = IMDBDataset(df_train['input_ids'].tolist(), df_train['label'].tolist())
test_dataset = IMDBDataset(df_test['input_ids'].tolist(), df_test['label'].tolist())

# Mô hình phân loại cảm xúc
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_sizes, activation_fn, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc_layers = nn.ModuleList()
        in_size = embed_dim
        for h in hidden_sizes:
            self.fc_layers.append(nn.Linear(in_size, h))
            in_size = h
        self.output = nn.Linear(in_size, 2)
        self.activation = activation_fn
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x).mean(dim=1)
        for layer in self.fc_layers:
            x = self.dropout(self.activation(layer(x)))
        return self.output(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cấu hình siêu tham số
configs = [
    {'batch_size': 32, 'learning_rate': 0.001, 'hidden_sizes': [128, 64], 'activation': 'relu'},
    {'batch_size': 32, 'learning_rate': 0.0005, 'hidden_sizes': [256, 128], 'activation': 'tanh'},
    {'batch_size': 64, 'learning_rate': 0.001, 'hidden_sizes': [256, 128, 64], 'activation': 'relu'},
    {'batch_size': 16, 'learning_rate': 0.0005, 'hidden_sizes': [512, 256], 'activation': 'tanh'},
    {'batch_size': 64, 'learning_rate': 0.01, 'hidden_sizes': [128, 64], 'activation': 'relu'},
]

# Hàm train + evaluate
def train_eval_model(config, num_runs=3):
    accuracies = []
    for run in range(num_runs):
        print(f"\nChạy lần {run+1} - Siêu tham số: {config}")
        activation_fn = getattr(F, config['activation'])
        model = SentimentModel(
            vocab_size=len(vocab),
            embed_dim=128,
            hidden_sizes=config['hidden_sizes'],
            activation_fn=activation_fn,
        ).to(device)

        optimizer = Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn=collate_fn)

        print(f"Đang huấn luyện với learning rate: {config['learning_rate']} và batch size: {config['batch_size']}")
        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

        print("Đánh giá mô hình trên tập kiểm tra:")
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x).argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(batch_y.tolist())
        acc = accuracy_score(all_labels, all_preds)
        print(f"  Độ chính xác (run {run+1}): {acc:.4f}")
        accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    return mean_acc, std_acc

# Chạy từng cấu hình
results = []
for i, config in enumerate(configs):
    print(f"\n Cấu hình {i+1}")
    mean_acc, std_acc = train_eval_model(config)
    results.append((i+1, mean_acc, std_acc))
    print(f"Kết quả trung bình: Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
