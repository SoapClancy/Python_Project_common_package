% This version of the objective function takes the Cholesky factor of the covariance matrix as an input..
function fVal = objectiveFunction(x_in,sorting_order,u,d,K,N) %#codegen

%%%INPUT: x_in = the parameter vector
%%%Output: fVal = objectve function vlaue
% K = number of Modes
% d = dimension
% N = Number of data points
% u = NXd cdf value matrix (This remains fixed in the entire optimization process)

% GETTING THE MU, SIGMA AND ALPHA FROM THE PARAMTER VECTOR X_IN
% Getting the paramter values related to the means of modes from x_in
% vector
mean_params = x_in(1:K*d);
% Reshaping it into a (KXd) matrix  
mu = reshape(mean_params,K,d);

% Removing these entries from the x_in vector
offSet1 = K*d;

% Getting the paramter values related to the choleskly factors of modes
num_V_params = (d+1)*d/2; % Number of params related to the cholesky factors 

V_mat = zeros(d,d,K);
Sigma = zeros(d,d,K);

temp_vec = cumsum(d:-1:1);

for k = 1:K
    % Getting choleskly factor parameters for the ith mode in the for loop.
    % Note: The first d elements in the vector corresponds to the diagonal
    % element and the remaining d*(d-1)/2 are the non-diagonal elements.
    current_mode_params = x_in(offSet1+(k-1)*num_V_params+1:offSet1+k*num_V_params);
    
    for j = 1:d
        V_mat(j:end,j,k) = current_mode_params(temp_vec(j)-d+j:temp_vec(j));
    end
    
    diag_elements = diag(V_mat(:,:,k));
    % Checking for the positive definiteness constraints. 
    if ~isempty(diag_elements(diag_elements<0))
        fVal = 1e50;
        return;
    end

    % Getting the covaraince matrix from a Cholskey factor
    Sigma(:,:,k) = V_mat(:,:,k)*V_mat(:,:,k)'; 
    
    clear current_mode_params diag_elements;
end

% % Removing the cholesky factor related entriers from the x_in vector
offSet2 = K*d + K*num_V_params; 

% Getting the Mixing weights.
alpha = [x_in(offSet2+1:end) 1-sum(x_in(offSet2+1:end))];

% Obtaining the inverseValues
y = computeInverseVals_vectorized(mu,Sigma,alpha,u,d,K,N,1);
for i = 1:d
   y(sorting_order(:,i),i) = y(:,i);
end


% EVALUATING THE OBJECTIVE FUNCTION
% Getting the first part of the obtjective function
small_mat = 1e-323*ones(N,1);

first_part = zeros(size(y,1),K);
y_hat = zeros(N,d,K);
for k = 1:K
    y_hat(:,:,k) = y - repmat(mu(k,:),N,1); % Getting the mean adjusted inverse vals
    temp_mat = y_hat(:,:,k)*(inv(V_mat(:,:,k)))';
    first_part(:,k) = alpha(k)*(1/(2*pi)^(d/2))*(1/prod(diag(V_mat(:,:,k))))*exp(-0.5*sum(temp_mat.*temp_mat,2));
    clear temp_mat;
end
first_part_ll = log(sum(first_part,2) + small_mat);  % A small positive number is added to avoid log(0)
clear y;

% Getting the second part of the objective function
second_part = zeros(N,K);
for j = 1:d
    temp_vector = zeros(N,K);
    for k = 1:K
        temp_vector(:,k) =  alpha(k)*(1/sqrt(2*pi*Sigma(j,j,k)))...
            *exp(-0.5*(1/Sigma(j,j,k))*(y_hat(:,j,k).^2));
    end
    second_part(:,j) = log(sum(temp_vector,2)+ small_mat);
end
second_part_ll = sum(second_part,2);
clear y_hat;

% Negative log likelihood is the objective function value
fVal = sum(-first_part_ll + second_part_ll);
