import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(1,100,500)
x_data /= 100
# print(x_data)
y = 3.1234 * x_data + 2.98
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 0.04
plt.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x distribusion')
plt.show()
plt.plot(x_data, y, 'r', linewidth=3)
# print(x_data)

x_data = np.linspace(0, 1, 500)
# print(x_data)
y_data = 3.1234 * x_data + 2.98 + np.random.randn(*x_data.shape) * 0.04
plt.scatter(x_data, y_data)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x distribusion')
plt.show()