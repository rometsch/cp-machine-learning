%% Compute data to train on.
% Use x(t) = x_0 + v_0 t + 0.5 g t^2
% and initial values
x_0 = 2;
v_0 = 8;
g = 9.81;
% Timestep and number of data points
N = 100;
tau = 1/(N-1); %1/(N-1);
% Construct actual data 
time = tau*[0:1:N-1];
pos = x_0 + v_0.*time - .5*g.*time.*time;
% Plot data.
% fig = figure;
% plot(time,pos,'b-');
% title('training data');
%% Train on data.
% Emmbed data in a linear feature space.
X = [pos(1:N-2); pos(2:N-1); ones(1,N-2)];
% and embed to centralized feature space.
mu = mean(pos);
Z = [];
Z(1,:) = X(1,:) - mu;
Z(2,:) = X(2,:) - mu;
Y = pos(3:N);
V = Y-mu;
% Calculate optimal feature vector using eqn. (7.2).
wT = (Y*transpose(X))*(X*transpose(X))^-1
% Estimate parameter vector for centralized coordinates.
uT = (V*transpose(Z))*(Z*transpose(Z))^-1
% Feature vector from analytic result.
wT_a = [-1, 2, -g*tau^2]
%% Use feature vector to make a prediction for new inital values.
x_0 = 5;
v_0 = 3;
% Calculate first 2 values.
Y_pred = [x_0, x_0 + v_0*tau - .5*g*tau^2];
Y_pred_a = [x_0, x_0 + v_0*tau - .5*g*tau^2];
V_pred = Y_pred - mu;
for i=3:N+1
    Y_pred(i) = wT*[Y_pred(i-2);Y_pred(i-1);1];
    Y_pred_a(i) = wT_a*[Y_pred_a(i-2);Y_pred_a(i-1);1];
    V_pred(i) = uT*[V_pred(i-2);V_pred(i-1)];
end
% Calculate ground truth.
Y_gt = x_0 + v_0.*time - .5*g.*time.*time;
% Plot prediction and ground truth.
% Only plot first K values.
K = round(N/2);
fig = figure(1); clf;
box
hold on;
plot(time(1:K),Y_gt(1:K),'Color','b','Marker','none');
plot(time(1:K),Y_pred(1:K),'Color','r','Marker','o','LineStyle','none');
plot(time(1:K),Y_pred_a(1:K),'Color',[0 0 0],'Marker','x','LineStyle','none');
hold off;
legend('ground truth','prediction trained vector','prediction analytic vector','location','southeast');
xlabel('t');
ylabel('x');
% fig = figure;
% hold on;
% plot(time(1:K),V(1:K),'b-');
% plot(time(1:K),V_pred(1:K),'r.');
% hold off;
% legend('ground truth','prediction trained vector','prediction analytic vector');