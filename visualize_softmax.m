function [] = visualize_softmax()
    W = load('saves/model-Softmax-Epochs-10-LearningRate-0.01-Layer-0.dat');
    width = 28;
    height = 28;
    for i = 1 : 10
        subplot(2, 5, i);
        image = reshape(W(:, i), width, height)';
        colormap jet;
        imagesc(image);
        title(num2str(i - 1));
    end
end