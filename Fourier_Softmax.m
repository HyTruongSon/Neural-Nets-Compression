function [] = Fourier_Softmax()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    
    W = load('saves/model-Softmax-Epochs-10-LearningRate-0.01-Layer-0.dat');
    
    %% Fourier transformation the Softmax
    acc_W = zeros(100, 1);
    F_W = fft(reshape(W, 784 * 10, 1));
    for percent = 1 : 100
        f_W = F_W;
        f_W(int32(size(F_W, 1) * percent / 100.0) + 1 : end) = 0.0; 
        W_fourier = reshape(ifft(f_W), 784, 10);
        test_predict = Softmax(W_fourier, test_images);
        acc_W(percent) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Frequency percent %d: Accuracy = %.4f\n', percent, acc_W(percent));
    end
    
    figure(1);
    plot(1:100, acc_W(:), 'r-+');
    xlabel('Frequency percent');
    ylabel('MNIST Testing Accuracy');
    title('Fourier transformation for the Softmax (Neural nets 784 x 10)');
end