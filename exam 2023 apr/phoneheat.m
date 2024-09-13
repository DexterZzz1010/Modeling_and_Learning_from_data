function [A,B,C,D,K,x0] = phoneheat(parameters,Ts)
% ODE file for phone heating
% parameters = [c1, ..., c4, g1, ..., g7] where
% c1-c4  - heat capacity parameters
% g1-g7 - heat conductance parameters, g1=1/R1, etc
% Ts = 0 gives continuous time estimation, assumed here

% Differential equation for the system is 
% E*dot(x) = AA*x + BB*u
% where E, AA and BB are linear functions of the parameters


E = diag(parameters(1:4));  % E = diag(c1,c2,c3,c4)
G = parameters(5:11);       % G = 1./R 

AA=[-G(1)-G(2)-G(3) G(2) G(3) 0 ; ...
    G(2) -G(2)-G(4) 0 0 ;...
    G(3) 0 -G(3)-G(5)-G(6) G(6);...
    0 0 G(6) -G(6)-G(7)];

BB=[G(1) 0 ;...
    0 G(4);...
    0 G(5);...
    0 G(7)];

A = inv(E)*AA;
B = inv(E)*BB;
C = eye(4);
D = zeros(4,2);
x0 = zeros(4,1);  % not used
K = zeros(4,4);   % not used

