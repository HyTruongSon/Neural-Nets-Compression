function [predict, RMSE] = Autoencoder(W, example)
    function [f] = sigmoid(x)
        f = 1.0 ./ (1.0 + exp(-x));
    end
    predict = sigmoid(sigmoid(example * W) * W');
    RMSE = sqrt(sum(sum((example - predict) .* (example - predict))) / (size(example, 1) * size(example, 2)));
end