%% problem 1 b)
% Reproduce time series from figure 7.2.
% Number of data points.
N = 100;
% Feature vectors.
%wT = [-4, 0, 0, 4, 0, 0];  %commend for use of external plotscript
% Inital conditions.
x1 = 9./13;
x2 = 3./7;
% Data vector.
x = [x1, x2];
Z = [];
% Construct the time series by iterating with the linear function.
for t=2:1:N-1
    z = [x(t)^2; x(t)*x(t-1); x(t-1)^2; x(t); x(t-1); 1];
    Z = [Z,z];
    x(t+1) = wT*z;
end
%% part c)
% Calculate estimated feature vector using martix form of eq (7.9).
% Be sure to construct matrices such that x_(n+1) corresponds to y_n.
Y = x(3:N);
wT_est = Y*transpose(Z)*(Z*transpose(Z))^-1;
% Use estimated feature vector to predict time series.
x_pred = [x1, x2];
for t=2:1:N
    z = [x_pred(t)^2; x_pred(t)*x_pred(t-1); x_pred(t-1)^2; x_pred(t); x_pred(t-1); 1];
    x_pred(t+1) = wT_est*z; 
end
%% part d)
% Construct new time series with random noise.
% Seed random number genreator.
rng(7,'twister');
epsilon = 0.01;
% Data vector.
xn = [x1, x2];
Zn = [];
% Construct the time series by iterating with the linear function.
for t=2:1:N-1
    z = [xn(t)^2; xn(t)*xn(t-1); xn(t-1)^2; xn(t); xn(t-1); 1];
    Zn = [Zn,z];
    xn(t+1) = mod(wT*z + epsilon*randn,1);
end
% Calculate estimated feature vector using martix form of eq (7.9).
% Be sure to construct matrices such that x_(n+1) corresponds to y_n.
Yn = xn(3:N);
wTn_est = Yn*transpose(Zn)*(Zn*transpose(Zn))^-1;
% Use estimated feature vector to predict time series.
xn_pred = [x1, x2];
for t=2:1:N
    z = [xn_pred(t)^2; xn_pred(t)*xn_pred(t-1); xn_pred(t-1)^2; xn_pred(t); xn_pred(t-1); 1];
    xn_pred(t+1) = wTn_est*z; 
end
