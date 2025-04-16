import numpy as np
import idx2numpy

# Các hàm kích hoạt
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Lớp DNN
class SimpleDNN:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.01):
        self.lr = learning_rate
        self.activation_name = activation
        
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Chọn hàm kích hoạt cho lớp ẩn
        if activation == 'sigmoid':
            self.act = sigmoid
            self.d_act = d_sigmoid
        elif activation == 'relu':
            self.act = relu
            self.d_act = d_relu
        elif activation == 'tanh':
            self.act = tanh
            self.d_act = d_tanh
        elif activation == 'leaky_relu':
            self.act = lambda x: leaky_relu(x, alpha=0.01)
            self.d_act = lambda x: d_leaky_relu(x, alpha=0.01)
        elif activation == 'softmax':
            self.act = None
            self.d_act = None
        else:
            raise ValueError("Hàm kích hoạt không hợp lệ. Dùng: relu, sigmoid, tanh, leaky_relu, softmax.")

    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.act(self.Z1) if self.act is not None else self.Z1
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true):
        m = y_true.shape[0]
        y_one_hot = np.zeros_like(self.A2)
        y_one_hot[np.arange(m), y_true] = 1

        dZ2 = self.A2 - y_one_hot
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.d_act(self.Z1) if self.d_act is not None else dA1
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, batch_size=32, epochs=10):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                self.forward(X_batch)
                self.backward(X_batch, y_batch)

            preds = self.forward(X)
            loss = -np.mean(np.log(preds[np.arange(len(y)), y] + 1e-9))
            acc = self.evaluate(X, y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {acc:.4f}")

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# Đọc và xử lý dữ liệu MNIST
train_images = idx2numpy.convert_from_file('./Week1/data/MNIST/raw/train-images-idx3-ubyte')
train_labels = idx2numpy.convert_from_file('./Week1/data/MNIST/raw/train-labels-idx1-ubyte')
test_images = idx2numpy.convert_from_file('./Week1/data/MNIST/raw/t10k-images-idx3-ubyte')
test_labels = idx2numpy.convert_from_file('./Week1/data/MNIST/raw/t10k-labels-idx1-ubyte')

X_train = train_images.reshape(-1, 28*28) / 255.0
X_test = test_images.reshape(-1, 28*28) / 255.0
y_train = train_labels
y_test = test_labels

# Các bộ siêu tham số thử nghiệm
param_sets = [
    (32, 0.1, 16, 'relu'),
    (16, 0.1, 64, 'sigmoid'),
    (64, 0.1, 32, 'tanh'), 
    (128, 0.1, 128, 'leaky_relu'), 
    (32, 0.1, 128, 'softmax')
]

# Chạy các bộ siêu tham số
for batch_size, learning_rate, hidden_size, activation in param_sets:
    print(f"\nĐang chạy với bộ siêu tham số: {batch_size}, {learning_rate}, {hidden_size}, {activation.upper()}")
    acc_list = []

    for run in range(5):
        print(f"\nRun {run+1}/5")
        model = SimpleDNN(input_size=784, hidden_size=hidden_size, output_size=10,
                          activation=activation, learning_rate=learning_rate)
        model.train(X_train, y_train, batch_size=batch_size, epochs=10)
        acc = model.evaluate(X_test, y_test)
        print(f"Test Accuracy: {acc:.4f}")
        acc_list.append(acc)

    mean_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    print(f"\nKết quả với activation: {activation}")
    print(f"Trung bình: {mean_acc:.4f}")
    print(f"Độ lệch chuẩn: {std_acc:.4f}")
