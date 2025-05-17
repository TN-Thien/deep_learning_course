import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lớp ResNet giống như trong script gốc
class ResNetForCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Load mô hình tốt nhất
model = ResNetForCIFAR10()
model.load_state_dict(torch.load("Week5/best_model_resnet18_cfg3_run1.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Nhãn của CIFAR-10
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Chuyển đổi ảnh
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

# Hàm phân loại ảnh
def classify_image(img):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)
    return {CIFAR10_LABELS[i]: float(probs[0][i]) for i in range(10)}

# Giao diện Gradio
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier (ResNet18)",
    description="Upload hoặc chụp một ảnh (kích thước nhỏ), mô hình sẽ dự đoán nhãn thuộc tập CIFAR-10."
)

if __name__ == "__main__":
    demo.launch()
