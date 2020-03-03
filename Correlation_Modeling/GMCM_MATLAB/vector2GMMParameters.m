function [mu,sigma,PComponents] = vector2GMMParameters(params,d,K)

% Defining the GMM object using the parameter vector
mu_params = params(1:K*d);
mu = reshape(mu_params,K,d);
offSet1 = K*d;

temp_vec = cumsum(d:-1:1);

num_V_params = (d+1)*d/2;
for k = 1:K
    current_mode_params = params(offSet1 + (k-1)*num_V_params+1 : offSet1 + k*num_V_params);
    V_mat = zeros(d,d);
    for j = 1:d
        V_mat(j:end,j) = current_mode_params(temp_vec(j)-d+j:temp_vec(j));
    end
    sigma(:,:,k) = V_mat*V_mat' + 0.01*eye(d);
    clear current_mode_params;
end

offSet2 = K*d + K*num_V_params;

PComponents = [params(offSet2+1:end) 1-sum(params(offSet2+1:end))];
