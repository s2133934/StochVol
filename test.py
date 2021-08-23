# http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/3-4-UnivarMLE.html

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# # Calculate the probability density function for values of x in [-6;6]
# x = np.linspace(start = -6, stop = 6, num = 200)
# #
# _ = plt.figure(num = 0, figsize = (10, 8))
# _ = plt.plot(x, norm.pdf(x, loc = 0, scale = 1), color = "black", 
#          label = "$\\mu=0$, $\sigma =1$")
# _ = plt.plot(x, norm.pdf(x, loc = 1, scale = 1.5), color = "red", 
#          label = "$\\mu=1$, $\sigma =1.5$")
# _ = plt.plot(x, norm.pdf(x, loc = -1, scale = 0.5), color = "blue", 
#          label = "$\\mu=-1$, $\sigma =0.5$")
# _ = plt.title("Probability density function of $N(\\mu, \\sigma^2)$")
# _ = plt.ylabel("Density")
# _ = plt.legend()
# plt.show()



from scipy.stats import chi2
# Calculate the probability density function for values of x in [0;10]
x = np.linspace(start = 0, stop = 10, num = 200)
#
_ = plt.figure(num = 1, figsize = (5, 4))
_ = plt.plot(x, chi2.pdf(x, df = 1), color = "black", label = "k = 1")
for i in range(2, 7):
    _ = plt.plot(x, chi2.pdf(x, df = i), label = "k = " + str(i))
_ = plt.ylim((0, 0.6))    
_ = plt.title("Probability density function of $\\chi^2_k$")
_ = plt.ylabel("Density")
_ = plt.legend()
plt.show()


