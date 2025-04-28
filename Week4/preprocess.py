import pandas as pd
import re
from underthesea import word_tokenize
import os
import json
from sklearn.preprocessing import LabelEncoder

# Đọc dữ liệu
df = pd.read_csv("Week4/data/Data_vnexpress.csv")

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', str(text))
    text = re.sub(r'[^A-Za-zÀ-ỹ0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def preprocess_text(text):
    cleaned = clean_text(text)
    tokenized = word_tokenize(cleaned, format="text")
    return tokenized

print("Đang xử lý dữ liệu...")

# Xử lý tiêu đề và nội dung
df['title'] = df['title'].astype(str).apply(preprocess_text)
df['content'] = df['content'].astype(str).apply(preprocess_text)

# Encode label từ text → số
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Lưu label mapping
label2id = {label: int(idx) for idx, label in enumerate(le.classes_)}
id2label = {int(v): k for k, v in label2id.items()}
with open("Week4/data/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

# Suffle dữ liệu
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Lưu lại file đã xử lý
output_path = "Week4/data/Data_vnexpress_processed.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Đã lưu dữ liệu đã xử lý vào: {output_path}")
print(f"Đã lưu label mapping vào: Week4/data/label_mapping.json")
