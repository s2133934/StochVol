# FUNCTION 1:

########## Test weak convergence of Euler-Maruyama ###########

# SDE is dX = mu*X dt + sigma*X dW,  X(0) = X_0
#      where mu = 2, sigma = 0.1, and X_0 = 1
#
# E-M uses 5 different timesteps: 2^(p-10),  p = 1,2,3,4,5.
# Examine weak convergence at T=1:  | E (X_L) - E (X(T)) |.
#
# Different paths are used for each E-M timestep.
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

np.random.seed(102)
mu=2
sigma=0.1
X_0=1
T=1
N=50000
num_powers = 6
Dtvals=np.power(float(2), [x-10 for x in range(1,num_powers)])

Xem=np.zeros((5,1))
for power in range(1,num_powers): # for all Dt values
    Dt = 2**(power-10)
    Xtemp = X_0*np.ones((N,1))

    for j in range(1,int(T/Dt)+1): #for each time along the path
        Winc = np.sqrt(Dt)*np.random.randn(N) #random increment
        Xtemp += Dt*mu*Xtemp + sigma*np.multiply(Xtemp.T,Winc).T
    Xem[power-1] = np.mean(Xtemp,0)
Xerr = np.abs(Xem - np.exp(mu))

plt.loglog(Dtvals,Xerr, 'o-', label='X errors')
plt.loglog(Dtvals,Dtvals, 'x-', label='(x=y)')
plt.axis([1e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$')
plt.ylabel('| $E(X(T))$ - Sample average of $X_L$ |')
plt.title('EM Weak convergence')
plt.legend()
plt.show()

### Least squares fit of error = C * Dt^q ###
A = np.column_stack((np.ones((p,1)), np.log(Dtvals)))
rhs=np.log(Xerr)
sol = np.linalg.lstsq(A,rhs,rcond=None)[0]
q=sol[1][0]
resid=np.linalg.norm(np.dot(A,sol) - rhs)
print('q = ', round(q,5))
print('residual = ', round(resid,5))

# Slope of the loglog to get convergence order beta
beta = (np.log(Xerr[-1]/Xerr[0])) / (np.log(Dtvals[-1]/Dtvals[0]))
print('Convergence order (weak): beta = ', np.mean(beta))

# FUNCTION 2

############ Test strong convergence of Euler-Maruyama ######################

# SDE is dX = mu*X dt + sigma*X dW,  X(0) = X_0
#      where mu = 2, sigma = 1, and X_0 = 1
#
# Discretized Brownian path over [0,1] has dt = 2^(-9).
# E-M uses 5 different timesteps: 16dt, 8dt, 4dt, 2dt, dt.
# Examine strong convergence at T=1:  E | X_L - X(T) |.
#
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

np.random.seed(100)

mu=2
sigma=1
X_0=1
T=1
N=2**9
dt = float(T)/N
M=1000 #

Xerr=np.zeros((M,5))
for s in range(M):
    dW=np.sqrt(dt)*np.random.randn(1,N)
    W=np.cumsum(dW)
    Xtrue = X_0*np.exp((mu-0.5*sigma**2)*T+sigma*W[-1])
    for p in range(5): #for each power
        R=2**p
        Dt=R*dt
        L=N/R
        Xem=X_0
        for j in range(1,int(L)+1):  # for each step in the path
            Winc=np.sum(dW[0][range(R*(j-1),R*j)])
            Xem += Dt*mu*Xem + sigma*Xem*Winc
        Xerr[s,p]=np.abs(Xem-Xtrue)

Dtvals=dt*(np.power(2,range(5)))
plt.loglog(Dtvals,np.mean(Xerr,0),'o-', label='Mean of errors')
plt.loglog(Dtvals,np.power(Dtvals,0.5),'x-',label='(x=y)')
plt.axis([1e-3, 1e-1, 1e-3, 10])
plt.xlabel('$\Delta t$')
plt.ylabel('Sample average of $|X(T)-X_L|$')
plt.title('emstrong.py',fontsize=16)
plt.legend()
plt.show()

### Least squares fit of error = C * Dt^q ###
A = np.column_stack((np.ones((5,1)), np.log(Dtvals)))
rhs=np.log(np.mean(Xerr,axis=0))
sol = np.linalg.lstsq(A,rhs,rcond=None)[0]
q=sol[1]
resid=np.linalg.norm(np.dot(A,sol) - rhs)
print('residual = ', round(resid,5))
print('q =', round(q,5))

# Slope of the loglog to get convergence order beta
beta = (np.log(Xerr[-1]/Xerr[0])) / (np.log(Dtvals[-1]/Dtvals[0]))
print('Convergence order (strong): beta = ', np.mean(beta))


# FUNCTION 3

########### Test strong convergence of Milstein #########################
#
# SDE is  dX = r*(K-X) dt + beta*X dW,   X(0) = Xzero,
#       where r = 2, K = 1, beta = 0.25, Xzero = 0.5.
#
# Discretized Brownian path over [0,1] has dt = 2^(-11).
# Milstein uses timesteps 128*dt, 64*dt, 32*dt, 16*dt (also dt for reference).
# Examines strong convergence at T=1:  E | X_L - X_T |.
# Adapted from 
# Desmond J. Higham "An Algorithmic Introduction to Numerical Simulation of 
#                    Stochastic Differential Equations"
# http://www.caam.rice.edu/~cox/stoch/dhigham.pdf

np.random.seed(100)
r=2; K=1; beta=0.25; Xzero=0.5
T=1; N=2**11; dt=float(T)/N
M=500
R = [1, 16, 32, 64, 128]

dW = np.sqrt(dt)*np.random.randn(M,N)
Xmil = np.zeros((M,5))
for p in range(5):
    Dt = R[p]*dt; L=float(N)/R[p]
    Xtemp=Xzero*np.ones(M)
    for j in range(1,int(L)+1):
        Winc=np.sum(dW[:,range(R[p]*(j-1),R[p]*j)],axis=1)
        Xtemp += r*(K-Xtemp)*Dt + beta*Xtemp*Winc + 0.5*beta**2*Xtemp*(np.power(Winc,2)-Dt)
    Xmil[:,p] = Xtemp

Xref = Xmil[:,0]
Xerr = np.abs(Xmil[:,range(1,5)] - np.tile(Xref,[4,1]).T)
Dtvals = np.multiply(dt,R[1:5])

plt.loglog(Dtvals,np.mean(Xerr,0),'o-')
plt.loglog(Dtvals,Dtvals,'x-')
plt.axis([5e-3, 1e-1, 1e-4, 1])
plt.xlabel('$\Delta t$'); plt.ylabel('Sample average of $|X(T)-X_L|$')
plt.title('milstrong.py',fontsize=16)

#### Least squares fit of error = C * Dt^q ####
A = np.column_stack((np.ones((4,1)), np.log(Dtvals)))
rhs=np.log(np.mean(Xerr,0))
sol = np.linalg.lstsq(A,rhs,rcond=None)[0]; q=sol[1]
resid=np.linalg.norm(np.dot(A,sol) - rhs)

plt.show()

# Slope of the loglog to get convergence order beta
beta = (np.log(Xerr[-1]/Xerr[0])) / (np.log(Dtvals[-1]/Dtvals[0]))
print('residual = ', round(resid,5))
print('q = ', round(q,5))
print('Convergence order (strong): beta = ', np.mean(beta))


