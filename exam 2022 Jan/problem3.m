load('problem3.mat')
%%
figure(1)
subplot(211)
plot(y)
subplot(212)
plot(u)

%% 剔除离群值
outliers_indices = [314, 628, 942];

% 计算相邻值的平均值
for i = 1:length(outliers_indices)
    index = outliers_indices(i);
    if index > 1 && index < length(y)
        % 计算相邻值的平均值
        average_value = (y(index - 1) + y(index + 1)) / 2;
        
        % 将离群值替换为相邻值的平均值
        y(index) = average_value;
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
%% 选择模型参数
% NN1 = struc(1:10,1:10,1:10); % tries 1000 different models
% V = arxstruc(ztrain,ztest,NN1);
% Nbest = selstruc(V)
% arxbest = arx(ztrain,Nbest)
%% 使用最佳参数生成模型
Nbest = [10 4 6];
arxbest = arx(ztrain,Nbest)
present(arxbest)
figure (1)
compare(ztest,arxbest,Inf)
figure (2)
fig4 = bodeplot(arxbest,{0.01,pi/h})
showConfidence(fig4,3)
figure (3)
resid(ztest,arxbest) %compare residual ()
%%
N = [2 2 4];
oe224 = oe(ztrain,N);
compare(ztest,oe224,arxbest,Inf)
fig = bodeplot(oe224,{0.01,pi/h})
showConfidence(fig,3)
resid(ztest,oe224)
present(oe224)
%%
s=tf('s');
sysc = 2*s/(1+s)/(1+2*s)*exp(-0.3*s);
sysd = c2d(sysc,h)
