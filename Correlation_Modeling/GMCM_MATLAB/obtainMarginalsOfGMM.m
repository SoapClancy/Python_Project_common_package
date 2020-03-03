%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%
% This function computes the univariate GMM object from the multivariate GMM params.
function marginalsGMM = obtainMarginalsOfGMM(mu,sigma,alpha,K,d)
    
for i = 1:d
    for j = 1:K
        mu_marginal(j,1)  = mu(j,i);
        sigma_marginal(:,:,j) = sigma(i,i,j);
        alpha_marginal(j) = alpha(j);
    end    
    marginalsGMM{i} = gmdistribution(mu_marginal,sigma_marginal,alpha_marginal); 
    clear mu_marginal sigma_marginal alpha_marginal;
end