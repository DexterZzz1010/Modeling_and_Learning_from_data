% Generates data for problem 6b on exam2024jan
% Assumes sigu = sige = 1;

close all ; clear 

% Some parameters corresponding to a stable system, no need to change this
a1 = 1.6;          
a2 = -0.7;
b = 2;

% Generate N data points, where correlation between u and e equals rho
N = 1000;                                % hint: increase to improve accuracy
rho = 0;                                 % try to change here
if abs(rho)>1 error("need abs(rho) <= 1"); end
u = randn(N,1);
e = rho*u + sqrt(1-rho^2)*randn(N,1);   % makes correlation = rho (can you see why?)

% Construct linear regression
X = zeros(N,3);
Y = zeros(N,1);
x(1) = 0;
x(2) = 0;
for t = 3:N
    x(t) = a1*x(t-1) + a2*x(t-2) + b*u(t) + e(t);
    X(t,:) = [x(t-1) x(t-2) u(t)];
    Y(t) = x(t);
end

% Solve normal equations to find parameter estimates
thetahat = X\Y

% Did we get correct estimates?
bias = thetahat - [a1;a2;b];