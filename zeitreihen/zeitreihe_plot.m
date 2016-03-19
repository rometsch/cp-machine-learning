clear
%% Calculate time series with different paramter vectors wT and plot them.
% first wT_1
wT = [-4, 0, 0, 4, 0, 0];
% Call skript.
zeitreihe
% run('timeseries.m');

% Plot time series.
fig = figure(1); clf;
subplot(2,3,[1,2]); box on;
hold on;
plot(x,'b');
plot(x_pred,'LineStyle','none','Marker','o','Color','red');
hold off;
title('w^T_1 without noise');
legend('time series','prediction');
xlabel('t'); ylabel('x_t');
xlim([0,100]);
% Plot parabola.
subplot(233); box on;
hold on;
scatter(x(1:1:N-1),x(2:1:N));
scatter(xn(1:1:N-1),xn(2:1:N));
hold off;
xlim([0,1]); ylim([0,1]);
xlabel('x_t'); ylabel('x_{t+1}');

% Plot time series with noise.
subplot(2,3,[4,5]); box on;
hold on;
plot(xn,'b');
plot(xn_pred,'LineStyle','none','Marker','o','Color','red');
hold off;
title('w^T_1 with noise');
legend('time series','prediction');
xlabel('t'); ylabel('x_t');
xlim([0,100]);
% Plot parabola.
subplot(236); box on;
hold on;
scatter(xn(1:1:N-1),xn(2:1:N));
scatter(xn_pred(1:1:N-1),xn_pred(2:1:N));
hold off;
xlim([0,1]); ylim([0,1]);
xlabel('x_t'); ylabel('x_{t+1}');
%% Do the same for wT_2
clear
wT = [0, 0, 4, 0, -4, 1];
% Call skript.
% run('timeseries.m');
zeitreihe

% Plot time series.
fig = figure(2); clf;
subplot(2,3,[1,2]); box on;
hold on;
plot(x,'b');
plot(x_pred,'LineStyle','none','Marker','o','Color','red');
hold off;
title('w^T_2 without noise');
legend('time series','prediction');
xlabel('t'); ylabel('x_t');
xlim([0,100]);
% Plot parabola.
subplot(233); box on;
hold on;
scatter(x(1:1:N-2),x(3:1:N));
scatter(x_pred(1:1:N-2),x_pred(3:1:N));
hold off;
xlim([0,1]); ylim([0,1]);
xlabel('x_t'); ylabel('x_{t+2}');

% Plot time series with noise.
subplot(2,3,[4,5]); box on;
hold on;
plot(xn,'b');
plot(xn_pred,'LineStyle','none','Marker','o','Color','red');
hold off;
title('w^T_2 with noise');
legend('time series','prediction');
xlabel('t'); ylabel('x_t');
xlim([0,100]);
% Plot parabola.
subplot(236); box on;
hold on;
scatter(xn(1:1:N-2),xn(3:1:N));
scatter(xn_pred(1:1:N-2),xn_pred(3:1:N));
hold off;
xlim([0,1]); ylim([0,1]);
xlabel('x_t'); ylabel('x_{t+2}');
