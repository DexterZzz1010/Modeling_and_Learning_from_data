clear
clc
run('init_quadcopter_model')
run('init_quadcopter_states')
run('init_noise_levels')
addpath ('models')
%%

Omega_H = sqrt((m*g)/(4*k)); %TODO
c = 1.5; %TODO
Omega2 = sqrt((2*(Omega_H^2))/(1+c^2)); %TODO
Omega4 = c*Omega2;
Omega_in.time = (0:inner_h:2)';
nbr_samples = length(Omega_in.time);
Omega_in.signals.values = zeros(nbr_samples,4);
segments = 10; %TODO
segment_size = floor(nbr_samples/segments)
switch_time = [floor(segment_size/2):segment_size:nbr_samples, nbr_samples]



Omega_in.signals.values(1:switch_time(1),:) = repmat([Omega_H Omega4 Omega_H Omega2],switch_time(1),1)
for i = 2:length(switch_time)-1
    seg_size = switch_time(i) - switch_time(i-1); %needed for boundary
    if mod(i,2) == 0
        Omega_in.signals.values(switch_time(i-1)+1:switch_time(i),:) = repmat([Omega_H Omega2 Omega_H Omega4],seg_size,1)
    else
        Omega_in.signals.values(switch_time(i-1)+1:switch_time(i),:) = repmat([Omega_H Omega4 Omega_H Omega2],seg_size,1)
    end
end

%%
disp('starting sim')
out = sim('omega_input')
disp('sim done')
%% Inspect Result
figure(1)
clf
subplot(3,1,1)
plot(out.eta)
title('Angles')
legend('\phi','\theta','\psi')
subplot(3,1,2)
plot(out.deta)
title('Angular Velocity')
legend('\phi','\theta','\psi')
subplot(3,1,3)
plot(out.p)
title('Position')
legend('x','y','z')
%%
Phidot = out.deta.data(:,1);
dat = iddata(Phidot,-out.Omega.data(:,2).^2 + out.Omega.data(:,4).^2,inner_h )
sys = procest(dat,'p0I')
figure(2)
clf
compare(dat,sys)
u=-Omega2^2+Omega4^2;
%I1_est = abs((k*l*u)/((max(Phidot)-min(Phidot))/(2/segments)))
I1_est = (k*l)/sys.kp 
true_I = I(1)