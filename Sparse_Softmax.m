function [] = Sparse_Softmax()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    
    W = load('saves/model-Softmax-Epochs-10-LearningRate-0.01-Layer-0.dat');
    
    %% Sparsify the Softmax layer
    acc_W = zeros(100, 1);
    weights = sort(reshape(W, 784 * 10, 1));
    for percent = 0 : 99
        threshold = weights(int32(size(weights, 1) * percent / 100.0) + 1);
        W_sparse = W;
        W_sparse(W < threshold) = 0.0;
        test_predict = Softmax(W_sparse, test_images);
        acc_W(percent + 1) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Sparsity percent %d: Accuracy = %.4f\n', percent, acc_W(percent + 1));
    end
    
    figure(1);
    plot(0:99, acc_W(:), 'r-+');
    xlabel('Sparsity percent');
    ylabel('MNIST Testing Accuracy');
    title('Sparsification of the Softmax layer (Neural nets 784 x 10)');
end