% 
% MATLAB Implementation of X-means
% 
% This code is a MATLAB implementation of the X-means originally implemented
% in Python by Y. GOTO. The original Python code can be found at:
% https://gist.github.com/yasaichi/254a060eff56a3b3b858
% 
% Reference:
% http://www.rd.dnc.ac.jp/~tunenori/doc/xmeans_euc.pdf
% 
% Author: N. Masuyama
% Date: 19 August, 2024
% Contact: masuyama@omu.ac.jp
% 
% License: This MATLAB code is licensed under the MIT License.
% 
% Tested on MATLAB R2024a
% 

rng(0);

% Data preparation
x = [normrnd(1, 0.1, [20, 1]); normrnd(2, 0.1, [20, 1]); normrnd(1, 0.1, [20, 1]); normrnd(2, 0.1, [20, 1])];
y = [normrnd(1, 0.1, [20, 1]); normrnd(1, 0.1, [20, 1]); normrnd(2, 0.1, [20, 1]); normrnd(2, 0.1, [20, 1])];
X = [x y];

% Perform clustering
x_means = XMeans().fit(X);

% Display results
% disp(x_means.labels_);
disp(x_means.cluster_centers_);
% disp(x_means.cluster_log_likelihoods_);
% disp(x_means.cluster_sizes_);

% Plot results
scatter(X(:, 1), X(:, 2), 30, x_means.labels_, 'filled');
hold on;
grid on;
axis square;
scatter(x_means.cluster_centers_(:, 1), x_means.cluster_centers_(:, 2), 100, 'r', '+');
xlim([0 3]);
ylim([0 3]);
legend('Data Points', 'Centroids');
hold off;
