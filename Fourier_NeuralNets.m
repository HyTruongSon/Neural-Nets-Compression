function [] = Fourier_NeuralNets()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W1 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-1.dat');
    
    %% Fourier transformation the first-layer of Neural Nets
    acc_W0 = zeros(100, 1);
    F_W0 = fft(reshape(W0, 784 * 256, 1));
    for percent = 1 : 100
        f_W0 = F_W0;
        f_W0(int32(size(F_W0, 1) * percent / 100.0) + 1 : end) = 0.0; 
        W_fourier = reshape(ifft(f_W0), 784, 256);
        test_predict = NeuralNets(W_fourier, W1, test_images);
        acc_W0(percent) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Frequency percent %d: Accuracy = %.4f\n', percent, acc_W0(percent));
    end
    
    %% Fourier transformation the second-layer of Neural Nets
    acc_W1 = zeros(100, 1);
    F_W1 = fft(reshape(W1, 256 * 10, 1));
    for percent = 1 : 100
        f_W1 = F_W1;
        f_W1(int32(size(F_W1, 1) * percent / 100.0) + 1 : end) = 0.0; 
        W_fourier = reshape(ifft(f_W1), 256, 10);
        test_predict = NeuralNets(W0, W_fourier, test_images);
        acc_W1(percent) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Frequency percent %d: Accuracy = %.4f\n', percent, acc_W1(percent));
    end
    
    figure(1);
    plot(1:100, acc_W0(:), 'r-+');
    hold on;
    plot(1:100, acc_W1(:), 'b-o');
    xlabel('Frequency percent');
    ylabel('MNIST Testing Accuracy');
    legend('First layer', 'Second layer');
    title('Fourier transformation (Neural nets 784 x 256 x 10)');
end