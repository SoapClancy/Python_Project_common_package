close all;
clear;

clc;
rng('shuffle') %Seeds the random number generator based on the current time

%% del
load('GMCM_(1, 4).mat')

del_idx = [3];
gmcObject_bestfit_params.mu(del_idx, :)=[];
gmcObject_bestfit_params.sigma(:, :, del_idx)=[];
gmcObject_bestfit_params.alpha(del_idx)=[];


gmcObject_bestfit_params.alpha = gmcObject_bestfit_params.alpha / sum(gmcObject_bestfit_params.alpha);

best = gmcdistribution(gmcObject_bestfit_params.mu, gmcObject_bestfit_params.sigma, gmcObject_bestfit_params.alpha);


sim=best.random(500000);
% figure;
% scatter(sim(:,1), sim(:,2));
% xlim([-0.02, 1.02]);
% ylim([-0.02, 1.02]);
% grid on
figure;
% scatplot(sim(:,1), sim(:,2),  'squares', [], [], [], 3, 8, colormap(jet));
scatter(sim(:,1), sim(:,2));
xlim([-0.02, 1.02]);
ylim([-0.02, 1.02]);
grid on

dscatter(sim(:,1), sim(:,2),'BINS',[100,100]);
colormap("jet");
xlim([-0.02, 1.02]);
ylim([-0.02, 1.02]);
grid on

