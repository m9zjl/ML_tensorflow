# a simple gradient descent code for biginners
# model = theta*x+b

import matplotlib.pyplot as plt
import numpy as np
theta = 4
b = 5
# linear regression and the result should be theta around 1 ,b around 0
x_ = [1, 2, 3, 4, 5, 6, 7]
y_ = [1, 2, 3, 4, 5, 6, 7]

alpha = 0.01
list0 = []
list1 = []
list2 = []
for i in range(1000):
    list0.append(i)
    list1.append(theta)
    list2.append(b)
    for x, y in zip(x_, y_):
        # to see why cost functino is written like this visite Andrew Ng`s video on coursera
        theta = theta - alpha * (theta * x + b - y) * x
        b = b - alpha * (theta * x + b - y)
    # if i % 100 == 0:
    #     print theta, b
plt.plot(np.array(list0), np.array(list1))
plt.plot(np.array(list0), np.array(list2))
plt.grid(True)
plt.show()
