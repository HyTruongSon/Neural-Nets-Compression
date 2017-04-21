function [predict] = Softmax(W, example)
    function [f] = sigmoid(x)
        f = 1.0 ./ (1.0 + exp(-x));
    end
    predict = sigmoid(sigmoid(example * W));
    [~, predict] = max(predict');
    predict = predict - 1;
    predict = predict';
end