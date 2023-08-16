clear all; close all; clc;
addpath(genpath('../Utils'))
addpath(genpath('../Utils'))
% metrics = ['corr', 'mse', 'psnr_fun', 'tv', 'wpsnr'];
% Calcular todas las métricas a la vez. Configurar.
metric = 'tv';

metric_fun = choose_metric(metric);

% Crea todos los thresholds posibles con esa dimension
threshold_size = 2; % 2, 3
thresholds = create_thresholds(threshold_size^2);

% Carga CIFAR100
CIFAR100 = load_cifar100();
height = size(CIFAR100, 2);
width = size(CIFAR100, 3);

% Means
metric_means = zeros(1, size(thresholds, 3));

% Crea el kernel para simular las convolucionales
% Establecer semilla
MyKernel = random_binary_matrix(3,3);

% Itera sobre todos los thresholds posibles
for i=1:size(thresholds, 3)
    
    metric_value = 0;
    
    %Elección de threshold
    threshold = thresholds(:,:,i);
    
    for j=1:size(CIFAR100, 1)
        
        % Carga de imagen individual
        input = rot90(rgb2gray(squeeze(CIFAR100(j,:,:,:))),-1);
        
        % Aplicar función Sign
        binary_input = binarize(input, 0.5);
        
        % Simulación de convolución binaria
        % Probar con diferentes tipos de padding
        binary_input = padarray(binary_input, [1 1], 'symmetric');
        SalWI       = conv2(binary_input, MyKernel, 'valid'); %%Filtering
        reluapp     = max(0,SalWI); %%Apply relu
        
        % Aplicación de threshold
        factor_h = height/threshold_size;
        factor_w = width/threshold_size;
        
        threshold_broadcast = repmat(threshold, factor_h, factor_w);
        proposed = reluapp - threshold_broadcast;
        
        output = binarize(proposed, 0);
        
        
        %Calcular métrica elegida
        if strcmp(metric, 'tv')
            metric_value = metric_value + metric_fun(output);
        else
            metric_value = metric_value + metric_fun(input, output);
        end
        
        figure(1)
        subplot(1,2,1)
        imagesc(input)
        colormap gray
        subplot(1,2,2)
        imagesc(output)
        colormap gray
        pause(1)
        
    end
    
    % Mostrar 5 con mejor puntaje y 5 con menor puntaje
    metric_means(1, i) = metric_value / size(CIFAR100, 1);
    
    
end