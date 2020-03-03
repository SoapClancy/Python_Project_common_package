function cdf_value=load_gmcm_and_cal_cdf(u, gmcm_model_file_)
rng('shuffle') %Seeds the random number generator based on the current time

load(gmcm_model_file_, 'gmcObject_bestfit_params');

gmcObject_bestfit = gmcdistribution(gmcObject_bestfit_params.mu, gmcObject_bestfit_params.sigma, gmcObject_bestfit_params.alpha);

cdf_value=gmcObject_bestfit.cdf(u);

end