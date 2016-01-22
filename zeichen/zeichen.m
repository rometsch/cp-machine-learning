%% MNIST-Classification
%% Load MNIST zeros and ones.
% Load 0's and 1's from training set.
X0 = loadMNISTnumber(0);
X1 = loadMNISTnumber(1);
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
% Plot code taken from M. Bethge.

figure(1), clf, colormap(gray)
subplot(221), imagesc(reshape(mu0,28,28)), axis('image','off')
title('class conditional mean for c=0')
subplot(222), imagesc(reshape(mu1,28,28)), axis('image','off')
title('class conditional mean for c=1')
subplot(223), imagesc(reshape(v0,28,28)), axis('image','off')
title('class conditional variances for c=0')
subplot(224), imagesc(reshape(v1,28,28)), axis('image','off')
title('class conditional variances for c=1')


%% b) Perform first PCA.
[Uj,Dj,Vj] = svd(cov(Xj'));
[U0,D0,V0] = svd(cov(X0'));
[U1,D1,V1] = svd(cov(X1'));


% Find last eigenvector greater 1.
evalj = diag(Dj), ind_lgj = find(evalj>1,1,'last');
evalj = evalj(1:ind_lgj);
eval0 = diag(D0), ind_lg0 = find(eval0>1,1,'last');
eval0 = eval0(1:ind_lg0);
eval1 = diag(D1), ind_lg1 = find(eval1>1,1,'last');
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
title('first 4 eigenvectors for c=0 data');
subplot(3,2,4), semilogy(eval0), xlim([1, 600]);
title('spectrum for c=0 data');
subplot(3,2,5),
mon = [reshape(U1(:,1),28,28),reshape(U1(:,2),28,28);reshape(U1(:,3),28,28),reshape(U1(:,4),28,28)];
imagesc(mon), axis('image','off');
title('first 4 eigenvectors for c=1 data');
subplot(3,2,6), semilogy(eval1), xlim([1, 600]);
title('spectrum for c=0 data');


%% Reduce dimension.
% Project data onto eigenvectors of joint data with eigenvalues > 1.
% Data in this space will be denoted by Z.
% Zj = Uj(:,1:ind_lgj)'*Xj;
Z0 = Uj(:,1:ind_lgj)'*X0;
Z1 = Uj(:,1:ind_lgj)'*X1;
Zj = [Z0,Z1];


%% Oriented PCA.

sigc= (size(Z0,2)*cov(Z0') + size(Z1,2)*cov(Z1'))/size(Zj,2);

% Scale data.
Zsj = sigc^(-.5)*Zj;
Zs0 = sigc^(-.5)*Z0;
Zs1 = sigc^(-.5)*Z1;

[Uc,Dc,Vc] = svd(Zsj*Zsj');

%% Reduce dimension further down to 2.;
Y1 = Uc(:,1:2)'*Zs1;
Yj = Uc(:,1:2)'*Zsj;


%% Plot the first two eigenvectors.

figure(3), clf, colormap(gray)
subplot(2,2,3);
hold on;
plot(Y0(1,:),Y0(2,:),'LineStyle','none','Marker','.');
plot(Y1(1,:),Y1(2,:),'LineStyle','none','Marker','.');
hold off;
xlabel('1st oriented PCA coefficient')
ylabel('2nd oriented PCA coefficient')
legend('c=0','c=1','Location','southeast');
subplot(2,2,1), imagesc(reshape(Uj(:,2),28,28)), axis('image','off'), colorbar;
title('2nd eigenvector of covariance matrix of joint data');
subplot(2,2,4), imagesc(reshape(Uj(:,1),28,28)), axis('image','off'), colorbar;
title('1st eigenvector of covariance matrix of joint data');

