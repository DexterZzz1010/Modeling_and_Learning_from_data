% A 1D model of a heated rod
% Note: The rod is not insulated at ends:
% Heat source u(t) is at x = 0 
% Heat transfers to surrounding room at x=L. Room has temperature zero

clear; close all

%% Parameters, dont change
h = 0.01;
N = 100;
Tmax = 100000*h;
k = 10*h;

%% Creating system, dont change
D2 = -2*eye(N) + diag(ones(N-1,1),1) + diag(ones(N-1,1),-1);
D2(1,1) = -1;
A = eye(N) + k*D2;
B = zeros(N,1);B(1)=k;


%% Chose sensor configuration, 1 or 3 sensors

nrsensors = 3;      % change between 1 and 3 here

location = 0.75; % 0.5 means that sensor location is in middle of rod
if nrsensors == 1
    C = zeros(1,N); 
    C(1,round(location*N)) = 1;   % Try moving location by changing above 
    D = 0;
elseif nrsensors == 3             % Dont change the sensor locations below
    C = zeros(nrsensors,N);
    C(1,round(0.2*N)) = 1;         
    C(2,round(0.50*N)) = 1;
    C(3,round(0.8*N)) = 1;
    D = zeros(nrsensors,1);
else
    error("nrsensors need to be 1 or 3")
end

G = ss(A,B,C,D,h);      % Original high order system before model reduction

%% Step response original high order model
figure(1)
[y,t,x] = step(G,0:h:Tmax);
plot(t,y,'b','linewidth',2);
hold on
set(gca,'fontsize',16)
grid on
%% Bode diagram original high order model
figure(2)
[mag1,phase1,w1] = bode(G,{1e-4 10});
mag1 = squeeze(mag1); phase1 = squeeze(phase1);
subplot(211)
loglog(w1,mag1,'b','linewidth',2)
hold on; grid on
subplot(212)
semilogx(w1,phase1,'b','linewidth',2)
hold on; grid on


%% Prepare for model reduction

[Gbal,sig] = balreal(G);

figure(3)
semilogy(sig,'-x')
set(gca,'fontsize',16)
axis([0 100 1e-16 1e2])
title('Singular values (sig)')
hold on
grid on


%% Do model reduction (two cases: either 1 or 3 sensors)

if nrsensors == 1
    levels  = [1e-1 1e-2 1e-4]  % How much error can we tolerate
elseif nrsensors == 3
    levels = [1e-3]                % Change here to something suitable
else
    error("nrsensors need to be 1 or 3")
end

for threshold = levels
    elim = (sig < threshold*sig(1));     % small entries of sig -> negligible states
    Ghat = modred(Gbal,elim);
    tf(Ghat)

    figure(1)
    [yh,t] = step(Ghat,0:h:Tmax);
    plot(t,yh,'--','linewidth',1.4);
    title('Step Response','fontsize',18)
    xlabel('Time (sec)','fontsize',18)
    if nrsensors == 1 % this legend can be improved...
      legend('G (order 100)','Ghat','Ghat','Ghat','Location','best')
    else
      legend('G (output 1)','G (output 2)','G (output 3)','Location','best')
    end

    figure(2)
    subplot(211)
    title('Bode diagram','fontsize',18)
    [mag2,phase2,w2] = bode(Ghat,w1);
    mag2 = squeeze(mag2); phase2 = squeeze(phase2);
    loglog(w2,mag2,'--','linewidth',1.4)
    axis([1e-4 10 1e-4 1e2])
    set(gca,'fontsize',16)
    subplot(212)
    if nrsensors==1
      semilogx(w2,phase2-phase2(1),'--','linewidth',1.4)
    else
      semilogx(w2,phase2-phase2(:,1),'--','linewidth',1.4)
    end
    axis([1e-4 10 -700 20])
    xlabel('Frequency (rad/s)','fontsize',18)
    set(gca,'fontsize',16)
    if nrsensors == 1 % this legend can be improved...
      legend('G (order 100)','Ghat','Ghat','Ghat','Location','best')
    else
      legend('G (output 1)','G (output 2)','G (output 3)','Location','best')
    end

end

%% evalfr(G,1) evaluates the dcgain of G
evalfr(G,1)
