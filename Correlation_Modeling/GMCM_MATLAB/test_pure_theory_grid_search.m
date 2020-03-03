close all;
clear;
clc;

size_=100;
mu_array = linspace(-10, 10, size_);
sigma = zeros(1, size_)+1;
alpha=zeros(1,size_)+ 1/size_;

%% GMM
x=-20:0.01:20;
y_pdf = zeros(size(x));
y_cdf = zeros(size(x));
for i = 1:size_
    pd = makedist('Normal','mu',mu_array(i),'sigma',sigma(i));
    y_pdf = y_pdf + alpha(i) * pdf(pd,x);
    y_cdf = y_cdf + alpha(i) * cdf(pd,x);
end

figure;
plot(x, y_pdf);
figure;
plot(x, y_cdf);
