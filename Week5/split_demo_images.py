import os
import pickle
import numpy as np
from PIL import Image

data_dir = "Week5/cifar-10-batches-py"
output_dir = "Week5/demo_images/cifar-10"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
    label_names = pickle.load(f, encoding='latin1')['label_names']

saved_classes = set()

for i in range(1, 6):
    file_path = os.path.join(data_dir, f"data_batch_{i}")
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        data = batch['data']
        labels = batch['labels']

        for j in range(len(labels)):
            label = labels[j]
            class_name = label_names[label]

            if class_name not in saved_classes:
                img_flat = data[j]
                img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
                img_pil = Image.fromarray(img)

                img_pil = img_pil.resize((256, 256), Image.NEAREST)

                save_path = os.path.join(output_dir, f"{class_name}.png")
                img_pil.save(save_path)

                saved_classes.add(class_name)

            if len(saved_classes) == 10:
                break
    if len(saved_classes) == 10:
        break

print("Đã lưu mỗi lớp một ảnh (256x256) vào thư mục:", output_dir)
