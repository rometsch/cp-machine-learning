%% MNIST-Classification
%% Load MNIST zeros and ones.
% Need MNIST data in folder 'MNIST'.
% See 'loadMNISTnumber.m' for exact required filenames.
num0 = 0;
num1 = 1;
% Load 0's and 1's from training set.
X0 = loadMNISTnumber('train',num0);
X1 = loadMNISTnumber('train',num1);
% Joint data matrix.
Xj = [X0, X1];


%% Compute sample means and covariance.
% Sample means.
mu0 = mean(X0,2);
mu1 = mean(X1,2);
% Variance.
v0 = var(X0,1,2);
v1 = var(X1,1,2);


%% Plot means and variance.

figure(1), clf, colormap(gray)
subplot(221), imagesc(reshape(mu0,28,28)), axis('image','off')
title(['class conditional mean for c=',num2str(num0)])
subplot(222), imagesc(reshape(mu1,28,28)), axis('image','off')
title(['class conditional mean for c=',num2str(num1)])
subplot(223), imagesc(reshape(v0,28,28)), axis('image','off')
title(['class conditional variances for c=',num2str(num0)])
subplot(224), imagesc(reshape(v1,28,28)), axis('image','off')
title(['class conditional variances for c=',num2str(num1)])
drawnow;


%% b) Perform first PCA.
[Uj,Dj,Vj] = svd(cov(Xj'));
[U0,D0,V0] = svd(cov(X0'));
[U1,D1,V1] = svd(cov(X1'));


% Find last eigenvector greater 1.
evalj = diag(Dj); ind_lgj = find(evalj>1,1,'last');
evalj = evalj(1:ind_lgj);
eval0 = diag(D0); ind_lg0 = find(eval0>1,1,'last');
eval0 = eval0(1:ind_lg0);
eval1 = diag(D1); ind_lg1 = find(eval1>1,1,'last');
eval1 = eval1(1:ind_lg1);


%% Plot eigenvectors and spectrums.
figure(2), clf, colormap(gray);
subplot(3,2,1),
mon = [reshape(Uj(:,1),28,28),reshape(Uj(:,2),28,28);reshape(Uj(:,3),28,28),reshape(Uj(:,4),28,28)];
imagesc(mon), axis('image','off');
title('first 4 eigenvectors for joint data');
subplot(3,2,2), semilogy(evalj), xlim([1, 600]);
title('spectrum for joint data');
subplot(3,2,3),
mon = [reshape(U0(:,1),28,28),reshape(U0(:,2),28,28);reshape(U0(:,3),28,28),reshape(U0(:,4),28,28)];
imagesc(mon), axis('image','off');
title(['first 4 eigenvectors for c=',num2str(num0),' data']);
subplot(3,2,4), semilogy(eval0), xlim([1, 600]);
title(['spectrum for c=',num2str(num0),' data']);
subplot(3,2,5),
mon = [reshape(U1(:,1),28,28),reshape(U1(:,2),28,28);reshape(U1(:,3),28,28),reshape(U1(:,4),28,28)];
imagesc(mon), axis('image','off');
title(['first 4 eigenvectors for c=',num2str(num1),' data']);
subplot(3,2,6), semilogy(eval1), xlim([1, 600]);
title(['spectrum for c=',num2str(num1),' data']);
drawnow;


%% Reduce dimension.
% Project data onto eigenvectors of joint data with eigenvalues > 1.
% Data in this space will be denoted by Z.
% Zj = Uj(:,1:ind_lgj)'*Xj;
Z0 = Uj(:,1:ind_lgj)'*X0;
Z1 = Uj(:,1:ind_lgj)'*X1;
Zj = [Z0,Z1];


%% Oriented PCA.

Sc= (size(Z0,2)*cov(Z0') + size(Z1,2)*cov(Z1'))/size(Zj,2);

% Scale data.
Zsj = Sc^(-.5)*Zj;
Zs0 = Sc^(-.5)*Z0;
Zs1 = Sc^(-.5)*Z1;

[Uc,Dc,Vc] = svd(Zsj*Zsj');

%% Reduce dimension further down to 2.;
Y0 = Uc(:,1:2)'*Zs0;
Y1 = Uc(:,1:2)'*Zs1;


%% Plot data with reduced dimensionality and first two eigenvectors of PCA of covariance matrix of joint data.

figure(3), clf;
subplot(2,2,3);
hold on;
plot(Y0(1,:),Y0(2,:),'LineStyle','none','Marker','.');
plot(Y1(1,:),Y1(2,:),'LineStyle','none','Marker','.');
hold off;
xlabel('1st oriented PCA coefficient');
ylabel('2nd oriented PCA coefficient');
legend(['c=',num2str(num0)],['c=',num2str(num1)],'Location','southeast');
subplot(2,2,1), imagesc(reshape(Uj(:,2),28,28)), axis('image','off'), colorbar;
title('2nd eigenvector of covariance matrix of joint data');
subplot(2,2,4), imagesc(reshape(Uj(:,1),28,28)), axis('image','off'), colorbar;
title('1st eigenvector of covariance matrix of joint data');
drawnow;


%% d) Discriminant values.

mu_Y0 = mean(Y0,2);
mu_Y1 = mean(Y1,2);

S_Y0 = cov(Y0');
S_Y1 = cov(Y1');


% Calculate coefficients for discriminant according to eq (7.15).
A = S_Y1^(-1) - S_Y0^(-1);
bT = - (mu_Y1'*S_Y1^(-1) - mu_Y0'*S_Y0^(-1) );

%% TODO: Check which formula is right!
gamma = mu_Y1'*S_Y1^(-1)*mu_Y1 - mu_Y0'*S_Y0^(-1)*mu_Y0 + sum(log(eval1(1:2))) - sum(log(eval0(1:2)));
% mu_Y1'*S_Y1^(-1)*mu_Y1 - mu_Y0'*S_Y0^(-1)*mu_Y0 + sum(log(eig(S_Y1)) - sum(log(eig(S_Y0)))
%%
d_Y0 = diag(Y0'*A*Y0)'+bT*Y0;
d_Y1 = diag(Y1'*A*Y1)'+bT*Y1;

%% e) Training error

ntheta = 1000;
theta = linspace(-80,100,ntheta);
% Add up errors from classification with theta.
e = sum(repmat(d_Y0,ntheta,1) < repmat(theta',1,length(d_Y0)),2);
e = e + sum(repmat(d_Y1,ntheta,1) > repmat(theta',1,length(d_Y1)),2);
% Normalize by total number of classified items.
e = e/(length(d_Y0)+length(d_Y1));
[e_min,ind] = min(e);
theta_best = theta(ind);
% Calculate optimal theta for gaussian variables using -gamma from eq
% (7.15).
theta_gauss = -gamma;

figure(4), clf;
hold on;
plot(theta,e,'LineStyle','none','Marker','.');
xlabel('threshold \theta'), ylabel('0-1 loss');
set(gca,'yscale','log');
l = ylim;
plot([theta_best,theta_best],[l(1),l(2)]);
plot([theta_gauss,theta_gauss],[l(1),l(2)]);
hold off;
legend('error','optimal \theta','Gauss \theta');
drawnow;


%% f) Test on test data.

Xt0 = loadMNISTnumber('test',num0);
Xt1 = loadMNISTnumber('test',num1);

Yt0 = Uc(:,1:2)'*Sc^(-.5)*Uj(:,1:ind_lgj)'*Xt0;
Yt1 = Uc(:,1:2)'*Sc^(-.5)*Uj(:,1:ind_lgj)'*Xt1;

% Compute discriminants.
d_Yt0 = diag(Yt0'*A*Yt0)'+bT*Yt0;
d_Yt1 = diag(Yt1'*A*Yt1)'+bT*Yt1;

% Plot discriminants.
figure(5), clf;
subplot(2,1,1);
hold on;
histogram(d_Y0,'DisplayStyle','stairs','Normalization','probability');
histogram(d_Y1,'DisplayStyle','stairs','Normalization','probability');
hold off;
title('discriminant values of training set');
xlabel('discriminant value'), ylabel('normalized count');
legend('c=0','c=1');

subplot(2,1,2);
hold on;
histogram(d_Yt0,'DisplayStyle','stairs','Normalization','probability');
histogram(d_Yt1,'DisplayStyle','stairs','Normalization','probability');
hold off;
title('discriminant values of test set');
xlabel('discriminant value'), ylabel('normalized count');
legend('c=0','c=1');
drawnow;

%% g) Compute 0-1-loss for test data.


% Add up errors from classification with theta.
et = (sum(d_Yt0 < theta_best) + sum(d_Yt1 > theta_best))/(length(d_Yt0)+length(d_Yt1))
egauss = (sum(d_Yt0 < theta_gauss) + sum(d_Yt1 > theta_gauss))/(length(d_Yt0)+length(d_Yt1))
