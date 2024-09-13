close all
clear

load sysid03.mat

%% 
y(1901:2100) = [];
u(1901:2100) = [];
N = length(y);
%% Plot data
figure(1)
subplot(211)
plot(y)
ylabel('y','FontSize',16)
grid on
subplot(212)
plot(u)
ylabel('u','FontSize',16)
grid on
shg
z = iddata(y,u,Ts);

indtrain = 1:2000;
indtest = 2001:N;
ztrain = iddata(y(indtrain), u(indtrain),Ts);
ztest = iddata(y(indtest), u(indtest), Ts);

%% And a step response
figure(2) 
subplot(211)
timev = (1:length(ystep))*Ts;
plot(timev, ystep)
ylabel('ystep','FontSize',16)
grid on
subplot(212)
plot(timev, ustep)
ylabel('ustep','FontSize',16)
grid on
shg

%%
% Code for trying different ARX model orders
% NN = [(1:15)'  (1:15)' ones(15,1)];
% V = arxstruc(z,z,NN);
% Nbest = selstruc(V);

% NN = [(1:15)'  (1:15)' ones(15,1)];
% V = oestruc(z,z,NN);
% Nbest = selstruc(V);

%% Estimating an ARX model with 10 parameters
arx511 = arx(ztrain,[15 15 1]);

M = arx511;
present(M)


%%
N = [6 6 1];
O = oe(z,N);
compare(ztest,O,M,Inf)
% resid(ztest,O)
present(O)
%%
figure(3)
pred_horizon = inf;
compare(ztest,M, pred_horizon)

%%
figure(4)
pred_horizon = inf;
compare(zstep,O,pred_horizon)

%%
figure(5)
subplot(211)
pwelch(y)
hold on
subplot(212)
pwelch(u)
hold on

%%
figure(6)
resid(ztest,O)

%% Bode diagram of estimated system
figure(7)
options = bodeoptions;
options.FreqUnits = 'Hz'; 
options.MagUnits = 'abs';
options.MagScale = 'log';
bode(O,{0.5,pi/Ts},options);
legend('O');
grid on

