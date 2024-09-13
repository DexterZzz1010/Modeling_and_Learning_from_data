close all
clear
load greydata.mat
odefun = 'phoneheat';
h = 0.2;
Ts = 0;

%% Plot experimental data
figure(1)
tvec = h*(0:1500)'; 
subplot(611), plot(tvec,data.y(:,1)), ylabel('y1')
subplot(612), plot(tvec,data.y(:,2)), ylabel('y2')
subplot(613), plot(tvec,data.y(:,3)), ylabel('y3')
subplot(614), plot(tvec,data.y(:,4)), ylabel('y4')
subplot(615), plot(tvec,data.u(:,1)), ylabel('u1')
subplot(616), plot(tvec,data.u(:,2)), ylabel('u2')
shg

%% Grey box identification

% Changing initial parameter guesses might help
nrpar = 11;
parameterguess = ones(1,nrpar);              % All parameters initialized to 1
m_init = idgrey(odefun, parameterguess,'c'); % Use continuous time estimation

% Freezing one of the parameters  
m_init.Structure.Parameters.Free(1,4) = 0;  %freeze 1 & 4 

% Restricting parameter ranges might help
m_init.Structure.Parameters.Minimum = zeros(1,nrpar);
m_init.Structure.Parameters.Maximum = 100*ones(1,nrpar);

% Will changing these opimization parameters help ?
opt2 = greyestOptions;
opt2.SearchOptions.MaxIterations = 100;
opt2.Focus = 'simulation';
opt2.InitialState = 'zero';
opt2.Display = 'on';

% Greybox identification done here
mhat = greyest(data,m_init,opt2);

% Study estimated parameters and their covariance
estimatedparameters = mhat.report.Parameters.ParVector
P = mhat.report.Parameters.FreeParCovariance;
par_stderror = sqrt(diag(P))
svdP = svd(P)


%% Study model performance

figure(2)
compare(data,mhat)

