```matlab
%%
% This evaluates different model orders
 
NN = [(1:15)'  (1:15)' ones(15,1)];
V = arxstruc(z,z,NN);
Nbest = selstruc(V);
```

```matlab
null(A,'rational')
```

```matlab
N = [2 2 4];
oe224 = oe(z,N);
compare(ztest,oe224,arxbest,Inf)
fig = bodeplot(oe224,{0.01,pi/h})
showConfidence(fig,3)
resid(ztest,oe224)
present(oe224)
```

```matlab
%% 剔除离群值
outliers_indices = [1.5,3.5,5.5,7.5,9.5];

% 计算相邻值的平均值
for i = 1:length(outliers_indices)
    index = outliers_indices(i);
    if index > 1 && index < length(y)
        % 计算相邻值的平均值
        average_value = (y(index*100 - 1) + y(index*100 + 1)) / 2;
        
        % 将离群值替换为相邻值的平均值
        y(index*100+1) = average_value;
    end
end
```

~~~matlab
%% 伯德图
figure(5)
bode(model_arx3,model_arx5,{0.1,pi/h})
legend('arx3','arx5')
grid on

%% spectral
figure(6)
window = 200;
pwelch(u,window)
hold on
pwelch(y,window)

%% residual
figure(7)
resid(z,model_arx3)

~~~

~~~matlab
%% 划分训练集与测试集
N = length(y);
ytrain = y(1:N/2);
utrain = u(1:N/2);
ytest = y(N/2+1:end);
utest = u(N/2+1:end);
ztrain = iddata(ytrain,utrain,h);
ztest = iddata(ytest,utest,h);
~~~

