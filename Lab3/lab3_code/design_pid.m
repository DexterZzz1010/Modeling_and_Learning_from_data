
%%
k_est = k;
I_est = I(1);

s=tf('s');
k1 = 1/I(1); 
k2 = -9.81;

%inner loop, angle control
om1 =  3;
zeta1 = 1;
kp1 = om1^2/k1;
kd1 = 2*zeta1*om1/k1;
N = 5;
T = 1/om1/N;
C1 = (kd1*s+kp1)/(1 + 1.4*s*T+s^2*T^2);

%Outer loop, position control -> Angle reference
Ti =15;
N = 15;
b = 0.4;
kk = 0.06;
C2 = (1+1/s/Ti)*(s+b)*10/(s+b*N)*kk;



% Z controller
zeta = 0.7
omega = 3
a =1
ctrl_z.Kd = m*(a+2*zeta*omega);
ctrl_z.Kp = m*(2*zeta*omega*a+omega^2);
ctrl_z.Ki = m*(a*omega^2);
