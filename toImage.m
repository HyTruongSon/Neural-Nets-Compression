function [image] = toImage(weight, height, width)
    image = reshape(weight, height, width);
    MIN = min(min(image));
    MAX = max(max(image));
    image = (image - MIN) / (MAX - MIN);
    image = image * 255.0;
    image = int32(image / 5.0) * 5;
    image = uint8(image);
end