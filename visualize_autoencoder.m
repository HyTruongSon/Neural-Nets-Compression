function [] = visualize_autoencoder()
    figure(1);
    W = load('saves/model-Autoencoder-nHidden-256.dat');
    width = 28;
    height = 28;
    for i = 1 : 36
        subplot(6, 6, i);
        image = toImage(W(:, i), height, width)';
        imshow(image);
    end
    
    figure(2);
    test_images = load('saves/test-images.dat');
    nImages = 5;
    for i = 1 : nImages
        image = test_images(i, :);
        reconstructed_image = Autoencoder(W, image);
        subplot(2, nImages, i);
        imshow(uint8(reshape(image, 28, 28) * 255)');
        title('Original');
        subplot(2, nImages, nImages + i);
        imshow(uint8(reshape(reconstructed_image, 28, 28) * 255)');
        title('Reconstruction');
    end
end