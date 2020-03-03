function params = gmmParameters2Vector(mu,sigma,alpha,K,d)

params = mu(1:K*d);

for k= 1:K
    temp_V_mat = chol(sigma(:,:,k))'; 
    for j = 1:d 
        params = [params temp_V_mat(j:d,j)'];
    end
    clear temp_V_mat;
end

params = [params alpha(1:end-1)];
