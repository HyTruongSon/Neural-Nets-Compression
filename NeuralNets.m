function [predict] = NeuralNets(W0, W1, example)
    function [f] = sigmoid(x)
        f = 1.0 ./ (1.0 + exp(-x));
    end
    predict = sigmoid(sigmoid(example * W0) * W1);
    [~, predict] = max(predict');
    predict = predict - 1;
    predict = predict';
end