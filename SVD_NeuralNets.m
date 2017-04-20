function [] = SVD_NeuralNets()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W1 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-1.dat');
    
    %% Singular Value Decomposition for W0
    fprintf('--- Singular Value Decomposition for W0 ----------------------\n');
    [U, D, V] = svd(W0);
    N = min(size(W0, 1), size(W0, 2));
    acc_W0 = zeros(N, 1);
    for k = N : -1 : 1
        U_k = U(:, 1:k);
        D_k = D(1:k, 1:k);
        V_k = V(:, 1:k);
        test_predict = NeuralNets(U_k * D_k * V_k', W1, test_images);
        acc_W0(k) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Rank %d: Accuracy = %.4f\n', k, acc_W0(k));
    end
    
    figure(1);
    plot(N:-1:1, acc_W0(N:-1:1), 'r-+');
    xlabel('Number of singular values');
    ylabel('MNIST Testing Accuracy');
    title('Low-rank approximation of the first layer (Neural nets 784 x 256 x 10)');
    
    %% Singular Value Decomposition for W1
    fprintf('--- Singular Value Decomposition for W1 ----------------------\n');
    [U, D, V] = svd(W1);
    N = min(size(W1, 1), size(W1, 2));
    acc_W1 = zeros(N, 1);
    for k = N : -1 : 1
        U_k = U(:, 1:k);
        D_k = D(1:k, 1:k);
        V_k = V(:, 1:k);
        test_predict = NeuralNets(W0, U_k * D_k * V_k', test_images);
        acc_W1(k) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Rank %d: Accuracy = %.4f\n', k, acc_W1(k));
    end
    
    figure(2);
    plot(N:-1:1, acc_W1(N:-1:1), 'b-o');
    xlabel('Number of singular values');
    ylabel('MNIST Testing Accuracy');
    title('Low-rank approximation of the second layer (Neural nets 784 x 256 x 10)');
end