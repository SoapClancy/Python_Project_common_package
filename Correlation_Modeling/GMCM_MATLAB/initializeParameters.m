function [x_init,bounds,A,b] = initializeParameters(d,K)

mu = rand(K,d);
sigma = eye(d);
alpha = rand(1,K);alpha = alpha/sum(alpha);
gmmObject = gmdistribution(mu,sigma(1:d,1:d,ones(1,K)),alpha);

mu_matrix = random(gmmObject,K);
x_init = mu_matrix(1:K*d);

for k = 1:K
    for j = 1:d 
        x_init = [x_init sigma(j:d,j)'];
    end
end

x_init = [x_init alpha(1:end-1)];


% Assigning Bounds to the means
lb = [];ub=[];
for i = 1:K
    lb = [lb;-inf(d,1)];
    ub = [ub;+inf(d,1)];
end

% Assigning Bounds to the Covariance matrix elements
diag_idx = 1:d+1:d^2;
lb_mat = -inf(d);
lb_mat(diag_idx) = 0.1;
lb_mat = tril(lb_mat);

% Defining the lower bounds
for k = 1:K
    for j = 1:d 
       lb= [lb;lb_mat(j:d,j)];
    end
end

% Defining the upper bounds
ub_mat = tril(inf(d));
for k = 1:K
    for j = 1:d 
       ub= [ub;ub_mat(j:d,j)];
    end
end

% Assigning Bounds to the mixing weights
lb = [lb;0.001*ones(K-1,1)];
ub = [ub;0.999*ones(K-1,1)];

bounds = [lb ub];

% Setting the Linear constraints
A = [zeros(1,numel(x_init)-K+1) ones(1,K-1)];
b = 0.999;
 
