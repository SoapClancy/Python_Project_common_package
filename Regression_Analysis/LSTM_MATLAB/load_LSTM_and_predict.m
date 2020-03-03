function YPred = load_LSTM_and_predict(X, LSTM_file_)
rng('shuffle') %Seeds the random number generator based on the current time

X = X';

load(LSTM_file_, 'net');

YPred = predict(net,X);

YPred = YPred';
end