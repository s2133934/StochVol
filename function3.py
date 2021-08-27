#MILSTRONG Test strong convergence of Milstein
#
# SDE is  dX = r*X*(K-X) dt + beta*X dW,   X(0) = Xzero,
#       where r = 2, K = 1, beta = 0.25, Xzero = 0.5.
#
# Discretized Brownian path over [0,1] has dt = 2^(-11).
# Milstein uses timesteps 128*dt, 64*dt, 32*dt, 16*dt (also dt for reference).
#
# Examines strong convergence at T=1:  E | X_L - X_T |.
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
#
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(100)))

r, K, T, N = 2, 1, 1, 2**11
beta=0.25
Xzero=0.5
dt=float(T)/N
M=500
R = [1, 16, 32, 64, 128]
num_vals = len(R)
dW = np.sqrt(dt)*rs.randn(M,N)
Xmil = np.zeros((M,num_vals))

for i in range(num_vals):
    Dt = R[i]*dt
    L=float(N)/R[i]
    Xtemp = Xzero*np.ones(M)
    Dtr = Dt * r
    for j in range(1,int(L)+1):
        Winc=np.sum(dW[:,range(R[i]*(j-1),R[i]*j)],axis=1)
        Xtemp += Dtr*Xtemp*(K-Xtemp) + beta*Xtemp*Winc \
                 + 0.5*beta**2*Xtemp*(np.power(Winc,2)-Dt)
    Xmil[:,i] = Xtemp

Xref = Xmil[:,0]
Xerr = np.abs(Xmil[:,range(1,num_vals)] - np.tile(Xref,[4,1]).T)
Dtvals = np.multiply(dt,R[1:5])

plt.loglog(Dtvals,np.mean(Xerr,0),'b*-')
plt.loglog(Dtvals,Dtvals,'r--')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$'); plt.ylabel('Sample average of $|X(T)-X_L|$')
plt.title('milstrong.py',fontsize=16)

#### Least squares fit of error = C * Dt^q ####
coeffMatrix = np.column_stack((np.ones((4,1)), np.log(Dtvals)))
xerr_log=np.log(np.mean(Xerr,0))
x_lsq = np.linalg.lstsq(coeffMatrix,xerr_log,rcond=None)[0]; q=x_lsq[1]
resid=np.linalg.norm(np.dot(coeffMatrix,x_lsq) - xerr_log)
print('residual = ', resid)
print('q = ', q)
plt.show()
