import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy.polynomial.polynomial import polyfit

ys = [ -8,  27, -12, -13,  16, -13,   3,  -6,  17,  27,  14,  60,  63,  57,
       39,  67,  76,  22,  80,  60,  62,  69,  73,  42,  87,  55,  58,  55,
       95, 115, 103, 115,  69,  79,  75, 101, 100, 102, 106, 114]

xs = np.array([i for i in range(len(ys))], dtype=np.float64)

correlation_coef = np.corrcoef(xs, ys)[0,1]

# How good of a fit is our best fit
determination_coef = correlation_coef ** 2
print(determination_coef)

b, m = polyfit(xs, ys, 1)

style.use('fivethirtyeight')
plt.plot(xs, ys, '.')
plt.plot(xs, b + m * xs)
plt.show()
