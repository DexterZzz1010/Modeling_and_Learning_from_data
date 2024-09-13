close all; clear
load systemiddata

u = z.InputData;
y = z.OutputData;
%%
figure(1)
time = (1:length(u))*h;
subplot(211)
plot(time,u)
subplot(212)
plot(time,y)
grid on
%%
% Use this prediction horizon in the compare function (1 second)
horizon = 5; 

% An initial ARX model with 6 parameters
figure(2)
arxmodel = arx(z,[3 3 1])
compare(z,arxmodel,horizon)

figure(3)
bode(arxmodel)

figure(4)
step(arxmodel)
%%
% Find a better Box-Jenkins model
% Hint nk = 1 for this system
% The model should have at most 6 parameters
% i.e. nb + nc + nd + nf <= 6
% 
% bjmodel = bj(z, [nb nc nd nf nk])

% Also explain why it is clear that the Box Jenkins model is better than
% the ARX model

% To save time, you do not have to split data into training and test sets


% Use this prediction horizon in the compare function (1 second)
horizon = 5; 

% An initial ARX model with 6 parameters
figure(2)
bjmodel  = bj(z,[1 1 2 1 1])
compare(z,bjmodel,horizon)

figure(3)
bode(bjmodel)

figure(4)
step(bjmodel)