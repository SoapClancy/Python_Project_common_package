function yi=interp1q_custom(x,y,xi)
%INTERP1Q Quick 1-D linear interpolation.
%   F=INTERP1Q(X,Y,XI) returns the value of the 1-D function Y at the points
%   of column vector XI using linear interpolation. Length(F)=length(XI).
%   The vector X specifies the coordinates of the underlying interval.
%   
%   If Y is a matrix, then the interpolation is performed for each column
%   of Y in which case F is length(XI)-by-size(Y,2).
%
%   NaN's are returned for values of XI outside the coordinates in X.
%
%   INTERP1Q is quicker than INTERP1 on non-uniformly spaced data because
%   it does no input checking. For INTERP1Q to work properly:
%   X must be a monotonically increasing column vector.
%   Y must be a column vector or matrix with length(X) rows.
%
%   Class support for inputs x, y, xi:
%      float: double, single
%
%   See also INTERP1.

%   Copyright 1984-2008 The MathWorks, Inc.
%   $Revision: 1.15.4.3 $  $Date: 2008/03/28 15:24:12 $

siz = size(xi);

%[xxi, k] = sort(xi);
k = (1:siz(1))';
[~, j] = sort([x;xi]);
r(j) = 1:length(j);
r = r(length(x)+1:end) - (1:length(xi));
r(k) = r;
r(xi==x(end)) = length(x)-1;
ind = find((r>0) & (r<length(x)));
ind = ind(:);
yi = NaN(length(xi),size(y,2),superiorfloat(x,y,xi));
rind = r(ind);
xrind = x(rind);
u = (xi(ind)-xrind)./(x(rind+1)-xrind);
yrind = y(rind,:);
yi(ind,:)=yrind + bsxfun(@times,y(rind+1,:)-yrind,u);

if min(size(yi)) == 1 && numel(xi) > 1
   yi = reshape(yi,siz); 
end
