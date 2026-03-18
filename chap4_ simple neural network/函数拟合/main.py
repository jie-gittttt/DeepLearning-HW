import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据采集与预处理
# ==========================================
# 设置随机种子以保证结果可复现
np.random.seed(42)

# 在 [-pi, pi] 之间生成 2000 个点
num_samples = 2000
X = np.linspace(-np.pi, np.pi, num_samples).reshape(-1, 1)
y = np.sin(X)

# 随机打乱数据集
indices = np.random.permutation(num_samples)
X_shuffled = X[indices]
y_shuffled = y[indices]

# 按 80% 训练集，20% 测试集进行划分
split_idx = int(num_samples * 0.8)
X_train, y_train = X_shuffled[:split_idx], y_shuffled[:split_idx]
X_test, y_test = X_shuffled[split_idx:], y_shuffled[split_idx:]

# ==========================================
# 2. 神经网络模型定义 (纯 NumPy 实现)
# ==========================================
# 定义网络结构参数
input_size = 1
hidden_size = 64  # 隐藏层神经元个数
output_size = 1

# 使用 He 初始化 (He Initialization) 来初始化权重，对 ReLU 激活函数效果更好
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
b2 = np.zeros((1, output_size))

# 定义 ReLU 激活函数及其导数
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# ==========================================
# 3. 模型训练
# ==========================================
learning_rate = 0.01
epochs = 10000
m_train = X_train.shape[0]

print("开始训练神经网络...")
for epoch in range(epochs):
    # --- 前向传播 (Forward Propagation) ---
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_pred = Z2

    # 计算均方误差 (MSE Loss)
    loss = np.mean((y_pred - y_train) ** 2)

    # --- 反向传播 (Backward Propagation) ---
    # 计算输出层的梯度
    dZ2 = 2.0 / m_train * (y_pred - y_train)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # 计算隐藏层的梯度
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # --- 参数更新 (Gradient Descent) ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # 每 1000 次迭代打印一次损失值
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} / {epochs} | Loss (MSE): {loss:.6f}")

print("训练完成！\n")

# ==========================================
# 4. 测试与拟合效果验证
# ==========================================
# 在测试集上进行前向传播
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
y_test_pred = np.dot(A1_test, W2) + b2

# 计算测试集误差
test_loss = np.mean((y_test_pred - y_test) ** 2)
print(f"测试集最终 Loss (MSE): {test_loss:.6f}")

# ==========================================
# 5. 可视化拟合效果 (需要 matplotlib)
# ==========================================
# 为了画出平滑的预测曲线，我们在 -pi 到 pi 生成连续的点进行预测
X_plot = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
Z1_plot = np.dot(X_plot, W1) + b1
A1_plot = relu(Z1_plot)
y_plot_pred = np.dot(A1_plot, W2) + b2

plt.figure(figsize=(10, 6))
# 绘制真实函数曲线
plt.plot(X_plot, np.sin(X_plot), label="True Function: y = sin(x)", color="green", linewidth=2)
# 绘制测试集上的预测散点
plt.scatter(X_test, y_test_pred, label="Neural Network Predictions", color="red", alpha=0.6, s=15)

plt.title("Two-layer ReLU Neural Network fitting y = sin(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()