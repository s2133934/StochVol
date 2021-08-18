library(yuima)

# Ex 1. (One-dimensional diffusion process)
# To describe
# dXt = -3*Xt*dt + (1/(1+Xt^2+t))dWt,
# we set
mod1 <- setModel(drift = "-3*x", diffusion = "1/(1+x^2+t)", solve.variable = c("x"))
# We may omit the solve.variable; then the default variable x is used
mod1 <- setModel(drift = "-3*x", diffusion = "1/(1+x^2+t)")
# Look at the model structure by...
str(mod1)

set.seed(123)
drift <- c("mu*x1", "kappa*(theta-x2)") 
diffusion <- matrix(c("c11*sqrt(x2)*x1", "0", "c12*sqrt(x2)*x1", "c22*epsilon*sqrt(x2)"),2,2) 
heston <- setModel(drift=drift, diffusion=diffusion, state.var=c("x1","x2"))
X <- simulate(heston, true.par=list(theta=0.5, mu=1.2, kappa=2, epsilon=0.2, c11=C[1,1], c12=C[1,2], c22=C[2,2]), xinit=c(100,0.5))
str(heston)
plot(X)


library(yuima)
ymodel <- setModel(drift="(2-theta2*x)", diffusion="(1+xˆ2)ˆtheta1") 
n <- 750
ysamp <- setSampling(Terminal = nˆ(1/3), n = n)
yuima <- setYuima(model = ymodel, sampling = ysamp)
set.seed(123)
yuima <- simulate(yuima, xinit = 1, 
                  true.parameter = list(theta1 = 0.2, theta2 = 0.3))

param.init <- list(theta2=0.5,theta1=0.5) 
low.par <- list(theta1=0, theta2=0) 
upp.par <- list(theta1=1, theta2=1)
mle1 <- qmle(yuima, start = param.init,lower = low.par, upper = upp.par)

summary(mle1)

