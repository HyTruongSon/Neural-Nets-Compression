function [] = Sparse_NeuralNets()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W1 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-1.dat');
    
    %% Sparsify the first-layer of Neural Nets
    acc_W0 = zeros(100, 1);
    weights = sort(reshape(W0, 784 * 256, 1));
    for percent = 0 : 99
        threshold = weights(int32(size(weights, 1) * percent / 100.0) + 1);
        W_sparse = W0;
        W_sparse(W0 < threshold) = 0.0;
        test_predict = NeuralNets(W_sparse, W1, test_images);
        acc_W0(percent + 1) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Sparsity percent %d: Accuracy = %.4f\n', percent, acc_W0(percent + 1));
    end
    
    figure(1);
    plot(0:99, acc_W0(:), 'r-+');
    xlabel('Sparsity percent');
    ylabel('MNIST Testing Accuracy');
    title('Sparsification of the first layer (Neural nets 784 x 256 x 10)');
    
    %% Sparsify the second-layer of Neural Nets
    acc_W1 = zeros(100, 1);
    weights = sort(reshape(W1, 256 * 10, 1));
    for percent = 0 : 99
        threshold = weights(int32(size(weights, 1) * percent / 100.0) + 1);
        W_sparse = W1;
        W_sparse(W1 < threshold) = 0.0;
        test_predict = NeuralNets(W0, W_sparse, test_images);
        acc_W1(percent + 1) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Sparsity percent %d: Accuracy = %.4f\n', percent, acc_W1(percent + 1));
    end
    
    figure(2);
    plot(0:99, acc_W1(:), 'b-o');
    xlabel('Sparsity percent');
    ylabel('MNIST Testing Accuracy');
    title('Sparsification of the second layer (Neural nets 784 x 256 x 10)');
    
    figure(3);
    plot(0:99, acc_W0(:), 'r-+');
    hold on;
    plot(0:99, acc_W1(:), 'b-o');
    xlabel('Sparsity percent');
    ylabel('MNIST Testing Accuracy');
    legend('First layer', 'Second layer');
    title('Sparsification (Neural nets 784 x 256 x 10)');
end