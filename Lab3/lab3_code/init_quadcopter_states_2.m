% Set the initial states of the system
p0 = [0,0,0]';
v0 = [0,0,0]';
q0 = [1,0,0,0]'; % Corresponds to euler angles of (0,0,0)
w0 = [0,0,0]';

assert(norm(q0) == 1, 'The initial quaternion q0 must be a unit vector');

% Defines the maximum angle in the randomized attitude setpoints
max_ang_ref = pi/8;

% Rotor speed saturations
omega_min_lim = 500;  % rad/s
omega_max_lim = 2500; % rad/s

% Heuristic saturations to maintain non-zero torques for largt thrusts
sat_lim = 0.1;
Tm = 4.*k.^2.*omega_min_lim;
Tp = 4.*k.*omega_max_lim.^2;
T_min_lim = Tm + (Tp - Tm) * sat_lim;
T_max_lim = Tp - (Tp - Tm) * sat_lim;
tau_max_lim = +1e-5;
tau_min_lim = -1e-5;

inner_h = 1e-2
sample_time=1/60