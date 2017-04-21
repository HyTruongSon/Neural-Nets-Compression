function [] = SVD_Softmax()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    
    W = load('saves/model-Softmax-Epochs-10-LearningRate-0.01-Layer-0.dat');
    
    %% Singular Value Decomposition
    fprintf('--- Singular Value Decomposition for Softmax Layer ----------------------\n');
    [U, D, V] = svd(W);
    N = min(size(W, 1), size(W, 2));
    acc_W = zeros(N, 1);
    for k = N : -1 : 1
        U_k = U(:, 1:k);
        D_k = D(1:k, 1:k);
        V_k = V(:, 1:k);
        test_predict = Softmax(U_k * D_k * V_k', test_images);
        acc_W(k) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Rank %d: Accuracy = %.4f\n', k, acc_W(k));
    end
    
    figure(1);
    plot(N:-1:1, acc_W(N:-1:1), 'r-+');
    xlabel('Number of singular values');
    ylabel('MNIST Testing Accuracy');
    title('Low-rank approximation of the Softmax layer (Softmax 784 x 10)');
end