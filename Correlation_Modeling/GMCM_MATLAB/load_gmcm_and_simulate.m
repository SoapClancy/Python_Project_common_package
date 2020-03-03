function simulated=load_gmcm_and_simulate(gmcm_model_file_, n)
rng('shuffle') %Seeds the random number generator based on the current time

load(gmcm_model_file_, 'gmcObject_bestfit_params');

gmcObject_bestfit = gmcdistribution(gmcObject_bestfit_params.mu, gmcObject_bestfit_params.sigma, gmcObject_bestfit_params.alpha);

simulated=gmcObject_bestfit.random(n);

end