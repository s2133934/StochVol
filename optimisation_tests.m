% https://uk.mathworks.com/help/matlab/ref/fminsearch.html#d123e428544
clear all; clc

% fun = @(x)100*(x(2) - x(1)^2)^2 + (1 - x(1))^2;
% x0 = [-1.2,1];
% x = fminsearch(fun,x0)

%%% Generate path:
%generate timebase
x_0 = 0.01;
T = 1.0;
N = 1000;
dt = T/N;

% Inverse sampling to get normal
unis =rand(N);
norms = norminv(unis);
check=normpdf(norms,0,1);
plot(norms,check,'.')

% generate path
k=0.1;
sigma = 0.1;
theta = 0.1;
X(1) = x_0;

for i=2:N
    X(i) = X(i-1) * exp(-k*dt) + theta*(1-exp(-k*dt)) + sigma*sqrt((1-exp(-2*k*dt))/(2*k))*norms(i-1);
end

t_grid(1) = 0;
for k=2:N
    t_grid(k) = k*dt;
end

plot(t_grid,X) %Shows the OU path generated

x_excel=[0.01, 0.01178935, 0.013438872, 0.008729023, 0.011549686, 0.009860681, 0.01256937, 0.009693261, 0.01543231, 0.014349636, 0.011756408, 0.010903393, 0.00650562, -0.001079047, -0.003765747];
X=x_excel;
N=length(x_excel);
dt=0.001;

%%% Write the MLE for OU
alpha = 0.01;
gammma = 0;
beta = -0.1;
sigma = 0.1;
summ=0;

for j=2:N
    term0 = (sqrt(2 * sigma ^ 2 * pi * dt) * (abs(X(j - 1)) ^ (gammma)));
    term1 = (2 * sigma ^ 2) * dt * (abs(X(j - 1)) ^ (2 * gammma));
    term2 = (X(j) - (X(j - 1) * (1 + beta * dt) + alpha * dt)) ^ 2;
    summ = summ + log(1 / term0) - (term2 * (1 / term1));
end

summ 

%optimise!

