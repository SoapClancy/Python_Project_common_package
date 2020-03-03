% Author: Ashutosh Tewari (tewaria@utrc.utc.com)
% Affiliation: Decision Support & Machine Intelligence
%              United Technologies Research Center
%              East Hartford, CT: 06118
% Code based on the paper 'Parametric Characterization of MultiModal Densities with non-Gaussian
% Nodes', A. Tewari, M.J. Giering, A. Raghunathan, OEDM workshop, ICDM 2011 (paper included in the package) 

% This gmcdistribution is modified by Mingzhe Zou: 
% 1) adding conditional probability functions for conditional CDF evaluation and conditional sampling purpose
% 2) adding Copula cdf_partial_derivative
% best = gmcdistribution(gmcObject_bestfit_params.mu, gmcObject_bestfit_params.sigma, gmcObject_bestfit_params.alpha);

% DEFINING GMC (Gaussian Mixture Copula) CLASS
classdef gmcdistribution
    
    properties (SetAccess = private)
        mu     % k x d mean vectors
        sigma  % d x d x k  covariance matrices
        alpha  % k x 1  mixing proportions
    end
    
    methods
        % CONSTRUCTOR METHOD
        % Constructs a gmcdistribution object for a given parameter set.
        function obj = gmcdistribution(mu,sigma,alpha)  
            obj.mu = mu;
            obj.sigma = sigma;
            obj.alpha = alpha;
        end
        
        % METHOD TO FIT A GMC DISTRIBUTION TO DATA
        % Inputs: u  :N x d matrix of CDF values obtained after fitting marginal densities.
        %       : K  : number of clusters
        %       : d  : number of dimensions
        %       : N  : number of data samples.
        % varargin:  'Start' :  'rand_init' OR 'EM_init'  (default'EM_init');
        %            'replicates' : Integer  (default 1)
        %            'iteration'  : Integer  (default 100)
        %            'algorithm'  : 'active-set' OR 'interior-point' (default 'active-set')
        % Example:
        % obj.fit(u,K,d,N,'Start','rand_init','replicates',20,'algorithm','interior-point')
        
        function obj = fit(obj,u,K,d,N,iteration, varargin)
           
            % parsing the input argument
            p = inputParser;
            p.addParameter('Start','EM_init');
            p.addParameter('replicates','1');
            p.addParameter('algorithm','active-set');
            p.parse(varargin{:});

            % Choosing the initialization method. the 'rand_init' option
            % randomly assigns the parameters. 'EM_init' option usually generate
            % better inital guess (refer to the paper mentioned above).
            if strcmp(p.Results.Start,'rand_init')
                [x_init,bounds,A,b] = initializeParameters(d,K);
            elseif strcmp(p.Results.Start,'EM_init')
                [x_init,bounds,A,b] = initializeParameters(d,K);
                [x_init,~] = gmcm_EM(u,K,d,x_init);
            end
            
            % Performing the nonlinear optimization
            [x_final,~] = gmcmOptimization(u,K,d,N,bounds,A,b,x_init,iteration,p.Results.algorithm,str2double(p.Results.replicates));
            
            % converting the paramters in vectorized form into arrays
            [mu,sigma,alpha] = vector2GMMParameters(x_final,d,K);
            
            % updating the propoerties of the gmcdistribution object
            obj.mu = mu;
            obj.sigma = sigma;
            obj.alpha = alpha;           
            
        end
        
        
        %METHOD TO SAMPLE FROM THE GMC DISTRIBUTION 
        % Inputs: obj = GMC object
        %           N = number of samples required;
        % Output: samples = N x d matrix where d is the data dimension
        function gmc_samples = random(obj,N)
            
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            % Sampling from the gmm object
            gmm_samples = random(gmmObject,N);
            
            % Obtaining the marginal distribution of the gmm.
            gmm_marginals = obtainMarginalsOfGMM(obj.mu,obj.sigma,obj.alpha,K,d);
            
            % samples from the gmc are nothing but the marginal cdf values
            % of the gmm samples.
            gmc_samples = nan(size(gmm_samples));
            for i=1:d
                gmc_samples(:,i) = cdf(gmm_marginals{i},gmm_samples(:,i));
            end
            
        end
        
        % METHOD TO CLUSTER THE DATA GIVEN A GMC OBJECT
        % Inputs: obj = GMC object
        %         u   = N x d data to be clustered
        % Output: idx = N x 1 vector of cluster indices.
        function idx = cluster(obj,u)
            
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            % Cluserting the inverse values using the gmm object.
            idx = cluster(gmmObject,inverseVals);
        end
        
        % METHOD TO COMPUTE THE PDF VALUES W.R.T THE GMC DISCTRIBUTION
        % OBJECT
        %Inputs:  obj = GMC object
        %           u = N x d data to be clustered
        %Output: pdf_vals = N x 1 vector of the pdf values.
        function pdf_vals = pdf(obj,u)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
                       
            % Obtaining the log-likelihood of the numerator
            small_mat = 1e-323*ones(N,1);
            first_part = zeros(size(inverseVals,1),K);
            inverseVals_hat = zeros(N,d,K);
            for k = 1:K
                V_mat = chol(obj.sigma(:,:,k))';
                inverseVals_hat(:,:,k) = inverseVals - repmat(obj.mu(k,:),N,1); % Getting the mean adjusted inverse vals
                temp_mat = inverseVals_hat(:,:,k)*(inv(V_mat))';
                first_part(:,k) = obj.alpha(k)*(1/(2*pi)^(d/2))*(1/prod(diag(V_mat)))*exp(-0.5*sum(temp_mat.*temp_mat,2));
                clear temp_mat;
            end
            first_part_ll = log(sum(first_part,2) + small_mat);  % A small positive number is added to avoid log(0)
            clear inverseVals;

            % Getting the log-likelihood of the denominator
            second_part = zeros(N,K);
            for j = 1:d
                temp_vector = zeros(N,K);
                for k = 1:K
                    temp_vector(:,k) =  obj.alpha(k)*(1/sqrt(2*pi*obj.sigma(j,j,k)))...
                        *exp(-0.5*(1/obj.sigma(j,j,k))*(inverseVals_hat(:,j,k).^2));
                end
                second_part(:,j) = log(sum(temp_vector,2)+ small_mat);
            end
            second_part_ll = sum(second_part,2);
            clear inverseVals_hat;
            
            log_likelihood = first_part_ll - second_part_ll;
            pdf_vals = exp(log_likelihood);
        
        end
        
        % METHOD TO COMPUTE THE CDF VALUES W.R.T THE GMC DISCTRIBUTION
        % OBJECT
        %Inputs:  obj = GMC object
        %           u = N x d data to be clustered
        %Output: cdf_vals = N x 1 vector of the pdf values.
        function cdf_vals = cdf(obj,u)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            cdf_vals = cdf(gmmObject,inverseVals);
            
        end
        
        function partial_derivative = cdf_partial_derivative(obj, u, pos)
            % Only two dims are supported
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);              
            known_x = inverseVals(:, 1);
            known_y = inverseVals(:, 2);
            
            partial_derivative=0;
            for i=1:K                        
                this_alpha = gmmObject.ComponentProportion(i);
                
                this_mu = gmmObject.mu(i, :);
                mu1 = this_mu(:,1);
                mu2 = this_mu(:,2);
                
                this_sigma = gmmObject.Sigma(:,:, i);
                sigma1 = sqrt(this_sigma(1,1));
                sigma2 = sqrt(this_sigma(2,2));
                rho = this_sigma(1,2) /(sigma1*sigma2);
                
                if pos == 1
                     mu_cnt = mu2+rho.*sigma2.*(known_x-mu1)./sigma1;
                     sigma_cnt = zeros(size(mu_cnt)) + sigma2.*sqrt(1-rho.^2);
                     partial_derivative = partial_derivative+this_alpha.*normcdf(known_y,mu_cnt,sigma_cnt);
                else
                     mu_cnt = mu1+rho.*sigma1.*(known_y-mu2)./sigma2;
                     sigma_cnt = zeros(size(mu_cnt)) + sigma1.*sqrt(1-rho.^2);
                     partial_derivative = partial_derivative+this_alpha.*normcdf(known_x,mu_cnt,sigma_cnt);
                end            
            end
        end
    
        function partial_derivative = cdf_partial_derivative2(obj, u, pos)
            % Only two dims are supported
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
                        
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            %% marginal
            % Obtaining the log-likelihood of the numerator
            small_mat = 1e-323*ones(N,1);
            inverseVals_hat = zeros(N,d,K);
            for k = 1:K
                inverseVals_hat(:,:,k) = inverseVals - repmat(obj.mu(k,:),N,1); % Getting the mean adjusted inverse vals
            end

            % Getting the log-likelihood of the denominator
            second_part = zeros(N,K);
            j = pos;
            temp_vector = zeros(N,K);
            for k = 1:K
                temp_vector(:,k) =  obj.alpha(k)*(1/sqrt(2*pi*obj.sigma(j,j,k)))...
                    *exp(-0.5*(1/obj.sigma(j,j,k))*(inverseVals_hat(:,j,k).^2));
            end
            second_part(:,j) = log(sum(temp_vector,2)+ small_mat);
            second_part_ll = exp(sum(second_part,2));

            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);              
            x = inverseVals(:, 1);
            y = inverseVals(:, 2);
            
            partial_derivative=0;
            for i=1:K                        
                this_alpha = gmmObject.ComponentProportion(i);
                
                this_mu = gmmObject.mu(i, :);
                mu1 = this_mu(:,1);
                mu2 = this_mu(:,2);
                
                this_sigma = gmmObject.Sigma(:,:, i);
                sigma1 = sqrt(this_sigma(1,1));
                sigma2 = sqrt(this_sigma(2,2));
                rho = this_sigma(1,2) /(sigma1*sigma2);
                
                u = (x-mu1)/sigma1;
                v = (y-mu2)/sigma2;
                if pos == 1
                    t = (v - rho.*u) / sqrt(1-rho^2);
                    before_int_ = (1/(sqrt(2*pi) * sigma1)) .* exp(-(u.^2)./2);
                else
                    t = (u - rho.*v) / sqrt(1-rho^2);
                    before_int_ = (1/(sqrt(2*pi) * sigma2)) .* exp(-(v.^2)./2);
                end
                partial_derivative = partial_derivative + this_alpha .*before_int_ .* 0.5 .* (1+erf(t/sqrt(2)));
            end
            	
            partial_derivative = partial_derivative./second_part_ll;
%             figure; plot(partial_derivative); title('hard partial derivative');
%             figure; plot(partial_derivative./second_part_ll); title('nmb''s derivative');
% 
%             cdf_upper = cdf(gmmObject, [inverseVals(:, 1)+eps*10000, inverseVals(:,2)]);
%             cdf_lower = cdf(gmmObject, inverseVals);
%             by_defination = (cdf_upper-cdf_lower) ./ (eps*10000);
%             figure; plot(by_defination); title('Partial derivative of Copula');
%             figure; plot(by_defination ./second_part_ll); title('Defination of cdf derivative 2');

        end
        function cdf_value = cdf_condition_value(obj,u)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            condition = inverseVals(1,2);
            conditional_gmmObject = conditional_GMM(gmmObject, condition);
            
            cdf_value = cdf(conditional_gmmObject, inverseVals(:, 1));
            accuracy_error = 1e-100;
            cdf_value(cdf_value>1-accuracy_error) = 1-accuracy_error;
            cdf_value(cdf_value<accuracy_error) = accuracy_error;
        end
        
        function cdf_LUT = cdf_condition_LUT(obj,u, bins)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            condition = inverseVals(:,2);
            conditional_gmmObject = conditional_GMM(gmmObject, condition);
            
            %% Set up inverseVals according to resol
            temp = linspace(0.00001, 0.999999, bins);
            temp = temp';
            temp_2 = zeros(size(temp)) + u(:, 2);
            temp = [temp, temp_2];
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,temp,d,K,bins,0);
            
            cdf_LUT(:, 2) = cdf(conditional_gmmObject, inverseVals(:,1));
            cdf_LUT(:, 1) = temp(:, 1);
            
            %% Revise
            accuracy_error = 1e-100;
            cdf_LUT(cdf_LUT(:,2)<=accuracy_error | cdf_LUT(:,2)>=1-accuracy_error, :) = [ ];
            cdf_LUT(1, :) = 0;      cdf_LUT(end, :) = 1;
            [~, unique_idx] = unique(cdf_LUT(:, 2));
            cdf_LUT = cdf_LUT(unique_idx, :);
%             cdf_LUT(1, 2) = 0.000001;      cdf_LUT(end, 2) = 0.999999;
        end
%%
        function conditional_gmmObject = cdf_condition_f(obj,u)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            condition = inverseVals(:,2);
            conditional_gmmObject = conditional_GMM(gmmObject, condition);
            
        end
        
%%
        function cdf_random = cdf_condition_random(obj,u ,number)
        
            K = size(obj.mu,1);
            d = size(obj.mu,2);
            N = size(u,1);
            
            % Obtaining the inverse values with respect to the GMM marginal
            % distributions
            inverseVals = computeInverseVals_vectorized(obj.mu,obj.sigma,obj.alpha,u,d,K,N,0);
            
            % Defining  the gmm object from which the gmc distribution is
            % derived
            gmmObject = gmdistribution(obj.mu,obj.sigma,obj.alpha);
            
            condition = inverseVals(:,2);
            conditional_gmmObject = conditional_GMM(gmmObject, condition);
            
            gmm_samples = random(conditional_gmmObject, number);
            % Obtaining the marginal distribution of the gmm.
            gmm_marginals = obtainMarginalsOfGMM(obj.mu,obj.sigma,obj.alpha,K,d);
            % samples from the gmc are nothing but the marginal cdf values
            % of the gmm samples.           
            cdf_random = cdf(gmm_marginals{1},gmm_samples(:,1));

        end
    end
end