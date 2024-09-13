clear wind
windfreq = 0.5/pi; %1 rad/s
w0_wind = 2*pi*windfreq; %1 rad/s
epsilon = 3e-2;
s = tf('s')
Gwindd = w0_wind.^2/(s^2+2*epsilon*w0_wind*s+w0_wind^2);
wind.num = w0_wind^2;
wind.denum = [1 2*epsilon*w0_wind w0_wind^2];
wind_power = m;
