function [] = KMeans_NeuralNets()
    %% Loading dataset
    % train_images = load('saves/train-images.dat');
    % train_labels = load('saves/train-labels.dat');
    test_images = load('saves/test-images.dat');
    test_labels = load('saves/test-labels.dat');
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    W1 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-1.dat');
        
    %% KMeans on the first-layer
    fprintf('KMeans on the first-layer\n');
    acc_W0 = zeros(8, 1);
    for i = 1 : 8
        nClusters = 2^i;
        W0_kmeans = load(['saves/W0-clusters-', num2str(nClusters), '.dat']);
        test_predict = NeuralNets(W0_kmeans, W1, test_images);
        acc_W0(i) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Number of bits %d: Accuracy = %.4f\n', i, acc_W0(i));
    end
    
    %% KMeans on the second-layer
    fprintf('KMeans on the second-layer\n');
    acc_W1 = zeros(8, 1);
    for i = 1 : 8
        nClusters = 2^i;
        W1_kmeans = load(['saves/W1-clusters-', num2str(nClusters), '.dat']);
        test_predict = NeuralNets(W0, W1_kmeans, test_images);
        acc_W1(i) = sum(test_predict == test_labels) / size(test_images, 1);
        fprintf('Number of bits %d: Accuracy = %.4f\n', i, acc_W1(i));
    end
    
    plot(1:8, acc_W0(:), 'r-+');
    hold on;
    plot(1:8, acc_W1(:), 'b-o');
    xlabel('Number of bits');
    ylabel('MNIST Testing Accuracy');
    legend('First layer', 'Second layer');
    title('KMeans - Quantization - Lossy compression (Neural nets 784 x 256 x 10)');
end