% Getting the inverse values of univariate gmm CDFs empirically

function inverseVals = computeInverseVals_vectorized(mu,Sigma,alpha,u,d,K,N,is_u_sorted)

bin_num = 1000;

mu_mat = zeros(1,d,K);
sigma_mat = zeros(1,d,K);
alpha_mat = zeros(1,d,K);
range_mat = zeros(bin_num,d);
for j = 1:d
    mu_mat(1,j,:) = reshape(mu(:,j),1,1,K);
    sigma_mat(1,j,:) = (2*Sigma(j,j,:)).^0.5;
    alpha_mat(1,j,:) = reshape(alpha,1,1,K)/2;
    min_val = min(mu(:,j))-8*(max(Sigma(j,j,:))^0.5);
    max_val = max(mu(:,j))+8*(max(Sigma(j,j,:))^0.5);
    range_mat(:,j) = (min_val:(max_val-min_val)/(bin_num-1):max_val)';
end
mu_mat = reshape(mu',1,d,K);

erf_val_x = erf((range_mat(:,:,ones(K,1)) - mu_mat(ones(bin_num,1),:,:))./sigma_mat(ones(bin_num,1),:,:));

uVals = sum(alpha_mat(ones(bin_num,1),:,:).*(ones(bin_num,d,K)+erf_val_x),3);

inverseVals = zeros(N,d);
for j = 1:d 
    if is_u_sorted
        inverseVals(:,j) = interp1q_custom(uVals(:,j),range_mat(:,j),u(:,j));
    else
        inverseVals(:,j) = interp1q(uVals(:,j),range_mat(:,j),u(:,j));
    end
end
