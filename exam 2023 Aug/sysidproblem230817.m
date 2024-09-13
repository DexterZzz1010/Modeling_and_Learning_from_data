close all
clear
s = tf('s');

load sysiddata230817


%% 剔除离群值
outliers_indices = [1.5,3.5,5.5,7.5,9.5];

% 计算相邻值的平均值
for i = 1:length(outliers_indices)
    index = outliers_indices(i);
    if index > 1 && index < length(y)
        % 计算相邻值的平均值
        average_value = (y(index*100 ) + y(index*100 + 2)) / 2;
        
        % 将离群值替换为相邻值的平均值
        y(index*100+1) = average_value;
    end
end

%% 划分训练集与测试集
N = length(y);
ytrain = y(1:N/2);
utrain = u(1:N/2);
ytest = y(N/2+1:end);
utest = u(N/2+1:end);
ztrain = iddata(ytrain,utrain,h);
ztest = iddata(ytest,utest,h);

z = iddata(y,u,h);
%%

figure(1)
subplot(211)
plot(t,y)
ylabel('y')
grid on
subplot(212)
plot(t,u)
ylabel('u')
xlabel('time')
grid on

%%
% This step response was taken on the real system
% Can be used as test data for model validation
% 

figure(2)
plot(tstep,ystep,'k')
title('Measured step response (to be used as test data)')
xlabel('time')


%%
% Try some models

% NN = [(1:15)'  (1:15)' ones(15,1)];
% V = arxstruc(z,z,NN);
% Nbest = selstruc(V);
%%
model_arx3 = arx(ztrain,[13,13,1]);
model_arx5 = arx(ztrain,[10 10 1]);
N = [2 2 1];
oe224 = oe(ztrain,N);

figure(4)
pred_horizon = inf;
compare(ztest, model_arx3,model_arx5,oe224,pred_horizon)


%%
figure(5)
bode(model_arx3,model_arx5,{0.1,pi/h})
legend('arx3','arx5')
grid on

%%
figure(6)
window = 200;
pwelch(u,window)
hold on
pwelch(y,window)

% 

figure(7)
resid(ztest,model_arx3)


%%
present(model_arx5)

%%
figure(8)
y3 = step(model_arx3,tstep);
y5 = step(model_arx5,tstep);
plot(tstep,y3,tstep,y5,tstep,ystep,'k','LineWidth',1.5)
grid on
legend('y3','y5','true system')









