# 08/06/2021 - RJD - Messing about.

import sdeint
import numpy as np
import matplotlib.pyplot as plt

################################################
# a = 1.0
# b = 0.8
# tspan = np.linspace(0.0, 5.0, 5001)
# x0 = 0.1

# def f(x, t):
# # the dt term    
#     return -(a + x*b**2)*(1 - x**2)

# def g(x, t):
# # the dWt term
#     return b*(1 - x**2)

###################################################
A = np.array([[-0.5, -2.0],[ 2.0, -1.0]])

B = np.diag([0.5, 0.5]) # diagonal, so independent driving Wiener processes

tspan = np.linspace(0.0, 10.0, 10001)
x0 = np.array([3.0, 3.0])

def f(x, t):
    return A.dot(x)

def g(x, t):
    return B

# #################################################

result = sdeint.itoint(f, g, x0, tspan)
result2 = sdeint.stratint(f, g, x0, tspan)
plt.plot(result)
plt.plot(result2)
plt.show()
# print('result === \n', type(result))