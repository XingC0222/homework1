import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(0)

# 生成数据
x = np.linspace(-np.pi, np.pi, 100)  # 范围为 -π 到 π，取100个样本点
y = np.sin(x)

# 随机生成五次项的泰勒展开式作为初始值
coefficients = np.random.rand(6)  # 5次多项式，0到5共6个系数

# 定义预测函数
def predict(x, coeffs):
    return np.polyval(coeffs[::-1], x)  # 使用 np.polyval 计算多项式值

# 定义损失函数 (均方误差)
def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 定义梯度下降算法
def gradient_descent(x, y, coeffs, learning_rate, iterations):
    m = len(x)
    losses = []

    for i in range(iterations):
        y_pred = predict(x, coeffs)
        loss = loss_function(y, y_pred)
        losses.append(loss)

        # 计算梯度
        gradients = np.zeros(len(coeffs))
        for j in range(len(coeffs)):
            coeffs[j] += 1e-5
            y_pred_plus = predict(x, coeffs)
            coeffs[j] -= 1e-5
            gradient = (loss_function(y, y_pred_plus) - loss) / 1e-5
            gradients[j] = gradient

        # 更新系数
        coeffs -= learning_rate * gradients

        # 每100次输出损失
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    return coeffs, losses

# 设置参数，学习率与迭代次数
learning_rate = 0.0001
iterations = 50000

# 执行梯度下降
final_coeffs, losses = gradient_descent(x, y, coefficients, learning_rate, iterations)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(x, y, label='sin(x)', color='blue')
plt.plot(x, predict(x, final_coeffs), label='Predicted Function', color='red')
plt.title('Comparison between sin(x) and Predicted Function')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-np.pi, np.pi)  # 设置x轴范围为 -π 到 π
plt.legend()
plt.grid()
plt.show()
