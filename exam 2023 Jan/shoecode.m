close all; clear

load shoedata.mat

% Plot  data
figure(1)
subplot(211)
plot(height,shoe,'bx')
xlabel('height [m]')
ylabel('shoe size')
hold on; grid on
subplot(212)
plot(leg,shoe,'bx')
xlabel('leglength [m]')
ylabel('shoe size')
hold on; grid on

% Fit linear lines to data
subplot(211); 
p1 = polyfit(height,shoe,1)
Ypred1 = polyval(p1,height);
plot(height,Ypred1,'r')
text(1.42,40,'shoe = 24.0 height - 5.0','color','red')

subplot(212); 
p2 = polyfit(leg,shoe,1)
Ypred2 = polyval(p2,leg);
plot(leg,Ypred2,'r')
text(0.66,40,'shoe = 51.2 leglength - 3.3','color','red')


%%
% Standard LS estimation
% Using height and leg data and estimate also a bias
% The resulting estimated theta seems strange, can you explain the problem?

Y = shoe; 
X1 = [height ones(N,1)];
X2 = [leg    ones(N,1)];
X = [height leg ones(N,1)];     % Using both height and leglength
theta = X\Y                     % Solution to normal equations
theta1 = X1\Y 
theta2 = X2\Y 

YpredLS = X*theta;
rmsLS = sqrt(mean((YpredLS-Y).^2))

YpredLS1 = X1*theta1;
rmsLS1 = sqrt(mean((YpredLS1-Y).^2))

YpredLS2 = X2*theta2;
rmsLS2 = sqrt(mean((YpredLS2-Y).^2))

%%
% Linear regression analysis
%  training N times on different subset of data
%  removing mean values from LS estimation, will estimate Y-mean(Y)
%  Can add back mean value in Ypred later

thetav = zeros(2,N); 
Ypred = zeros(N,1);
for k = 1:N
    ind = [1:k-1 k+1:N];
    ms = mean(shoe(ind));
    mh = mean(height(ind));
    ml = mean(leg(ind));
    Yk = shoe(ind)-ms;
    Xk = [height(ind)-mh leg(ind)-ml];
    gamma = 0;      % what does this parameter do ?
    thetahat = inv(Xk'*Xk + gamma*eye(2))*Xk'*Yk;  
    Ypred(k) = ms + [height(k)-mh leg(k)-ml] * thetahat;
    thetav(:,k) = thetahat;
end

% Uncomment this to study results

rmseLOOCV  = sqrt(mean((Ypred-Y).^2))      % Leave-one-out CV performance
%rmse1 = sqrt(mean((Ypred1-Y).^2))     
%rmse2 = sqrt(mean((Ypred2-Y).^2))

% 
figure(3)
subplot(211)
hist(thetav(1,:),40)
title('histogram of theta1 (cv=200)')
subplot(212)
hist(thetav(2,:),40)
title('histogram of theta2 (cv=200)')
    
% Hint: Any ideas from the results of this ?
svd(Xk'*Xk)

figure(5)
plot(thetav(1,:), thetav(2,:),'x')


    