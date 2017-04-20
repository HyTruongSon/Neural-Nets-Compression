function [] = SVD_Autoencoder()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    W = load('saves/model-Autoencoder-nHidden-256.dat');
    
    %% Singular Value Decomposition for W
    fprintf('--- Singular Value Decomposition for W ----------------------\n');
    [U, D, V] = svd(W);
    N = min(size(W, 1), size(W, 2));
    RMSE_W = zeros(N, 1);
    for k = N : -1 : 1
        U_k = U(:, 1:k);
        D_k = D(1:k, 1:k);
        V_k = V(:, 1:k);
        W_k = U_k * D_k * V_k';
        [~, RMSE_W(k)] = Autoencoder(W_k, test_images);
        fprintf('Rank %d: RMSE = %.4f\n', k, RMSE_W(k));
    end
    
    figure(1);
    plot(N:-1:1, RMSE_W(N:-1:1), 'r-+');
    xlabel('Number of singular values');
    ylabel('MNIST Testing Reconstruction RMSE');
    title('Low-rank approximation (Autoencoder 784 x 256 x 784)');    
end