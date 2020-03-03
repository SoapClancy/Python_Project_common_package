function estimate_gmcm_and_save(u, gmcm_model_file_, K, iterations)
rng('shuffle') %Seeds the random number generator based on the current time

[N,d] = size(u);

gmcObject = gmcdistribution([],[],[]);
gmcObject_bestfit = gmcObject.fit(u,K,d,N,iterations);

gmcObject_bestfit_params.mu = gmcObject_bestfit.mu;
gmcObject_bestfit_params.sigma = gmcObject_bestfit.sigma;
gmcObject_bestfit_params.alpha = gmcObject_bestfit.alpha;

% save gmcm_model_file_ gmcObject_bestfit;
save(gmcm_model_file_, 'gmcObject_bestfit_params');
end