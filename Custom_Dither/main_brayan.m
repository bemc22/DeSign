clear all; close all; clc;
addpath(genpath('../Utils'))
addpath(genpath('Metrics'))
addpath(genpath('../Data'))

total_images = 100;

% Establecer semilla
rng(6)

% Crea todos los thresholds posibles con esa dimension
threshold_size = 2; % 2, 3
thresholds = create_thresholds(threshold_size^2);

% Means
metric_means = zeros(5, size(thresholds, 3));

% Crea el kernel para simular las convolucionales
MyKernel = random_binary_matrix(3,3);

%Decidir si usar padding o no
use_padding = 0;

%Tipo de padding a usar en convolución binario:
% ['symmetric', 'circular', 'replicate']
padding = 'symmetric';

% Carga CIFAR100
CIFAR100 = load_cifar100();
height = size(CIFAR100, 2)-2*not(use_padding);
width = size(CIFAR100, 3)-2*not(use_padding);

% Itera sobre todos los thresholds posibles
for i=1:size(thresholds, 3)
    i
    
    corr_value = 0;
    mse_value = 0;
    psnr_value = 0;
    tv_value = 0;
    wpsnr_value = 0;
    
    
    %Elección de threshold
    threshold = thresholds(:,:,i);
    
    for j=1:total_images
        
        % Carga de imagen individual
        input = rot90(rgb2gray(squeeze(CIFAR100(j,:,:,:))),-1);
        
        % Aplicar función Sign
        binary_input = binarize(input, 0.5);
        
        % Simulación de convolución binaria
        if use_padding
            binary_input = padarray(binary_input, [1 1], padding);
        else
            input = input(2:end-1,2:end-1);
        end
        
        SalWI       = conv2(binary_input, MyKernel, 'valid'); %%Filtering
        reluapp     = max(0,SalWI); %%Apply relu
        
        % Aplicación de threshold
        factor_h = height/threshold_size;
        factor_w = width/threshold_size;
       
        threshold_broadcast = repmat(threshold, factor_h, factor_w);
        proposed = reluapp - threshold_broadcast;
        
        % Salida final
        % Binariza a 0 y 1
        output = binarize_pos(proposed, 0);
        output = output .* threshold_broadcast;
        output  = (output  / 9)*0.99 + 0.01;
        reluapp = reluapp / 9;
        
        
        % Calculo de métricas
        % ['corr', 'mse', 'psnr_fun', 'tv', 'wpsnr']
        corr_value = corr_value + corr(reluapp, output);
        mse_value = mse_value + mse(reluapp, output);
        psnr_value = psnr_value + psnr_fun(reluapp, output);
        tv_value = tv_value + tv(output);
        wpsnr_value = wpsnr_value + wpsnr(reluapp, output);
        
        
%         figure(1)
%         subplot(1,2,1)
%         imagesc(input)
%         colormap gray
%         subplot(1,2,2)
%         imagesc(output)
%         colormap gray
%         pause(1)
        
    end
    
    % Almacenamiento de las 5 métricas para cada threshold
    % Mostrar 5 con mejor puntaje y 5 con menor puntaje
    metric_means(1, i) = corr_value / total_images;
    metric_means(2, i) = mse_value / total_images;
    metric_means(3, i) = psnr_value / total_images;
    metric_means(4, i) = tv_value / total_images;
    metric_means(5, i) = wpsnr_value / total_images;
     
    
end

save('metrics.mat', 'metric_means');


show_best_and_worst(metric_means, thresholds)





