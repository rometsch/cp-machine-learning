%% Compute data to train on.
% Use x(t) = x_0 + v_0 t + 0.5 g t^2
% and initial values
x_0 = 2;
v_0 = 8;
g = 9.81;
% Timestep and number of data points
N = 100;
tau = 2/133; %1/(N-1);
% Construct actual data 
time = tau*[1:1:N];
pos = x_0 + v_0.*time - .5*g.*time.*time;
% Plot data.
% fig = figure;
% plot(time,pos,'b-');
% title('training data');
%% Train on data.
% Emmbed data in a linear feature space.
X = [time(1:N-2); time(2:N-1); ones(1,N-2)];
Y = pos(3:N);
% Calculate optimal feature vector using eqn. (7.2).
wT = (Y*transpose(X))*pinv(X*transpose(X))
% Feature vector from analytic result.
wT_a = [-1, 2, -g*tau^2]
%% Use feature vector to make a prediction for new inital values.
x_0 = 5;
v_0 = 3;
% Calculate first 2 values.
Y_pred = [x_0, x_0 + v_0*tau - .5*g*tau^2];
Y_pred_a = [x_0, x_0 + v_0*tau - .5*g*tau^2];
for i=3:N+1
    Y_pred(i) = wT*[Y_pred(i-2);Y_pred(i-1);1];
    Y_pred_a(i) = wT_a*[Y_pred_a(i-2);Y_pred_a(i-1);1];
end
% Calculate ground truth.
Y_gt = x_0 + v_0.*time - .5*g.*time.*time;
% Plot prediction and ground truth.
% Only plot first K values.
K = round(N/2);
fig = figure;
hold on
plot(time(1:K),Y_gt(1:K),'b-');
plot(time(1:K),Y_pred(1:K),'r.');
plot(time(1:K),Y_pred_a(1:K),'g.');
legend('ground truth','prediction trained vector','prediction analytic vector');