close all;
clear;
clc;

load('x_train.mat')
load('y_train.mat');
y_train = y_train';

Y=load_LSTM_and_predict(x_train, 'training_1.mat');

ii = 120;
for i = ii : ii+20
figure;
plot(y_train(i,:)); hold on;
plot(Y(i',:));
end