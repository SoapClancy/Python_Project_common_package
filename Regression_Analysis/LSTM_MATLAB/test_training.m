close all;
clear;
clc;

load('x_train.mat')
load('x_validation.mat')
load('y_train.mat')
load('y_validation.mat')
file_ = 'train_LSTM_test_results.mat';

train_LSTM_and_save(x_train, y_train, x_validation, y_validation, file_);