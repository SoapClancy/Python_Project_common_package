function [mu,covMat] = updateCovMat(likelihood,givenData,regularization_param)

if nargin < 3
    regularization_param = 0;
end
    

dimension = size(givenData,2);

w = likelihood;
ws = w/sum(w);

mu = sum(repmat(ws,1,dimension).*givenData);

meanAdjustedData = givenData-repmat(mu,size(givenData,1),1);
meanAdjustedData = repmat(ws.^0.5,1,dimension).*meanAdjustedData;
covMat = meanAdjustedData'*meanAdjustedData + eye(dimension)*regularization_param;

