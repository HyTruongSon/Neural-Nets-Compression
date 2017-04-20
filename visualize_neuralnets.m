function [] = visualize_neuralnets()
    figure(1);
    W0 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-0.dat');
    width = 28;
    height = 28;
    for i = 1 : 36
        subplot(6, 6, i);
        % image = reshape(W0(:, i), width, height)';
        % colormap jet;
        % imagesc(image);
        image = toImage(W0(:, i), height, width)';
        imshow(image);
    end
    
    figure(2);
    W1 = load('saves/model-nHidden-256-Epochs-10-LearningRate-0.01-Layer-1.dat');
    width = 16;
    height = 16;
    for i = 1 : 10
        subplot(2, 5, i);
        image = reshape(W1(:, i), width, height)';
        colormap jet;
        imagesc(image);
        title(num2str(i - 1));
    end
end