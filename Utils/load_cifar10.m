function [images2, original] = load_cifar10(batch)
    % Loads Cifar10 Dataset

    images = load(strcat('Data/cifar-10-batches-mat/data_batch_',num2str(batch),'.mat'));
    images = double(images.data);
    images2 = zeros(10000,32,32);
    original = zeros(10000,32,32);
    
    for i=1:size(images, 1)
        
        image = squeeze(images(i,:))./255;
        image = rgb2gray(reshape(image, 32, 32, 3))';
        %figure;
        %imagesc(image)
        %colormap gray
        original(i,:,:) = image;
        image(image>0.5) = 1;
        image(image<=0.5) = -1;
        images2(i,:,:) = image;
        %figure;
        %imagesc(image)
        %colormap gray
        %close all
    
    end



end