import numpy as np
import matplotlib.pyplot as plt

# 生成x值
x = np.linspace(0, 1, 100)

# y = x
y1 = x

# y = x^0.5
y2 = np.sqrt(x)

# 绘制图像
plt.plot(x, y1, label='y=x')
plt.plot(x, y2, label='y=x^0.5')

# 添加标签和图例
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graphs of y=x and y=x^0.5')
plt.legend()

# 显示图像
plt.show()
