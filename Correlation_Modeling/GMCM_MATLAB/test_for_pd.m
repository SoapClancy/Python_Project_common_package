close all;
clear;
clc;
rng('shuffle') %Seeds the random number generator based on the current time

pseudo_zero = eps;

load('GMCM_(1, 3).mat')
best = gmcdistribution(gmcObject_bestfit_params.mu, gmcObject_bestfit_params.sigma, gmcObject_bestfit_params.alpha);

%% Simulatation
% sim=best.random(100000);
% figure;
% scatter(sim(:,1),sim(:,2))

%% CDF numerical partial derivative
% u = [linspace(0.0, 1, 5000)', zeros(5000,1) + 0.0050];
% u_plus = [u(:, 1)+ eps*100, u(:,2) ];
% u_minus =  [u(:, 1)- eps*100, u(:,2) ];
% pos = 1;

u = [zeros(10000, 1) + 0.55,  linspace(pseudo_zero, 1-pseudo_zero, 10000)'];
u_plus = [u(:, 1), u(:,2) + eps*100];
u_minus = [u(:, 1), u(:,2) - eps*100];
pos = 2;


cdf_ = best.cdf(u);
cdf_plus = best.cdf(u_plus);
cdf_minus = best.cdf(u_minus);

cdf_derivative = (cdf_plus - cdf_minus) ./ (2*eps*100);
figure;plot(u(:, pos), cdf_); title('CDF')

figure;plot(u(:, pos), cdf_derivative); title('Defination of cdf derivative'); hold on;


xx=best.cdf_partial_derivative2(u, pos);
plot(u(:, pos), xx);  title('pdf partial integal')
