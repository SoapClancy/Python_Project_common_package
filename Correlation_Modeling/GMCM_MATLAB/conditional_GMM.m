function conditional_gmmObject = conditional_GMM(gmmObject, condition)
%% Version: 2018.10.10
%% Input: gmmObject: Gaussian Mixture Object; a: unknow variable vector; b: conditional variable vector

size_b = size(condition,2);
size_a = gmmObject.NumVariables - size_b;
mu_a_C_b = nan(length(gmmObject.mu), 1);
sigma_a_C_b = nan(1, 1, length(gmmObject.Sigma));
weights_a_C_b = nan(length(gmmObject.mu), 1);

for k = 1 : gmmObject.NumComponents
    k_component_mu = gmmObject.mu(k, :);
    k_component_sigma = gmmObject.Sigma(:, :, k);
    k_mu_a_C_b = k_component_mu(1:size_a) + ...
        k_component_sigma(1:size_a, (size_a+1):end) / ...
        k_component_sigma((size_a+1):end, (size_a+1):end) *...
        (condition - k_component_mu((size_a+1):end));
    mu_a_C_b(k, :) = k_mu_a_C_b;
    
    k_sigma_a_C_b = k_component_sigma(1:size_a, 1:size_a) - ...
        k_component_sigma(1:size_a, (size_a+1):end) / ...
        k_component_sigma((size_a+1):end, (size_a+1):end) *...
        k_component_sigma((size_a+1):end, 1:size_a);
    sigma_a_C_b(:, :, k) = k_sigma_a_C_b;
    
    weights_a_C_b(k, :) = gmmObject.ComponentProportion(k) * pdf('Normal', condition, gmmObject.mu(k, (size_a+1):end), ...
                                                                                                                                                sqrt(gmmObject.Sigma((size_a+1):end, (size_a+1):end, k)));
end
weights_a_C_b = weights_a_C_b ./ sum(weights_a_C_b);
weights_a_C_b(abs(weights_a_C_b-0)<=0.0000000001) = 0.0000000001;
conditional_gmmObject = gmdistribution(mu_a_C_b, (sigma_a_C_b), weights_a_C_b);

end
