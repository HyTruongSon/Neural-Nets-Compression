function [] = Fourier_Autoencoder()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    
    W = load('saves/model-Autoencoder-nHidden-256.dat');
    
    %% Fourier transformation the Autoencoder
    RMSE = zeros(100, 1);
    F_W = fft(reshape(W, 784 * 256, 1));
    for percent = 1 : 100
        f_W = F_W;
        f_W(int32(size(F_W, 1) * percent / 100.0) + 1 : end) = 0.0; 
        W_fourier = reshape(ifft(f_W), 784, 256);
        [~, RMSE(percent)] = Autoencoder(W_fourier, test_images);
        fprintf('Frequency percent %d: Accuracy = %.4f\n', percent, RMSE(percent));
    end
    
    figure(1);
    plot(1:100, RMSE(:), 'r-+');
    xlabel('Frequency percent');
    ylabel('MNIST Testing Reconstruction RMSE');
    title('Fourier transformation for the Autoencoder (Neural nets 784 x 256 x 784)');
end