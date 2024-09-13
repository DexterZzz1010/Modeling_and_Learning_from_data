run('init_quadcopter_model')
run('init_quadcopter_states_2')
addpath ('models')
%%
run('design_pid')
%%
y_ref = 0.1
disp('starting sim')
out = sim('closed_loop','StopTime', '10')
disp('done sim')
%%
figure(1)
clf
subplot(2,1,1)
plot(out.p)
title('Position')
legend('x','y','z')
subplot(2,1,2)
plot(out.eta)
title('Angles')
legend('\phi','\theta','\psi')