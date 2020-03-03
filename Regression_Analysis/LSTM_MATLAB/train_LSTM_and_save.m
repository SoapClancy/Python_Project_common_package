function train_LSTM_and_save(x_train, y_train, x_validation, y_validation , LSTM_file_, MaxEpochs)
rng('shuffle') %Seeds the random number generator based on the current time

%% Define LSTM Network Architecture
x_train = x_train';
y_train = y_train';

x_validation = x_validation';
y_validation = y_validation';

save('x_train.mat', 'x_train')
save('y_train.mat', 'y_train')
save('x_validation.mat', 'x_validation')
save('y_validation.mat', 'y_validation')


numFeatures = size(x_train, 1);
numResponses = size(y_train, 1);
numHiddenUnits = 512;

layers = [ ...
    sequenceInputLayer(numFeatures)
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer
    bilstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',MaxEpochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.8, ...
    'ValidationData',{x_validation,y_validation}, ...
    'ValidationFrequency',25, ...
    'L2Regularization',0.0005, ...
    'Verbose',0, ...
    'Plots','training-progress');

%% Train and save
net = trainNetwork(x_train, y_train, layers, options);
save(LSTM_file_, 'net');
end