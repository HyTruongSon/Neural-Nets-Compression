function [] = Sparse_Autoencoder()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    
    W = load('saves/model-Autoencoder-nHidden-256.dat');
    
    %% Sparsify the Autoencoder
    RMSE_W = zeros(100, 1);
    weights = sort(reshape(W, 784 * 256, 1));
    for percent = 0 : 99
        threshold = weights(int32(size(weights, 1) * percent / 100.0) + 1);
        W_sparse = W;
        W_sparse(W < threshold) = 0.0;
        [~, RMSE] = Autoencoder(W_sparse, test_images);
        RMSE_W(percent + 1) = RMSE;
        fprintf('Sparsity percent %d: RMSE = %.4f\n', percent, RMSE_W(percent + 1));
    end
    
    figure(1);
    plot(0:99, RMSE_W(:), 'r-+');
    xlabel('Sparsity percent');
    ylabel('MNIST Testing Reconstruction RMSE');
    title('Sparsification of the Autoencoder (Neural nets 784 x 256 x 784)');
end