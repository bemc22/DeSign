clear all
close all

images = load_cifar100();

normalizacion = 1;

media = [0.507 0.487 0.441];
desv = [0.267 0.256 0.276];

if normalizacion
    for can = 1:3
        images(:,:,:,can) = (images(:,:,:,can)-media(can))/desv(can);
    end
end

kernel = 3;

M = [];

for pruebas = 1:1
    T =[];
    cont = 0;
    pruebas
    
    for c = 1:3
        for i = 1:size(images,1)
            
            KernelSize  = 3; %%kernel size
            MyImage     = rot90(squeeze(images(i,:,:,c)), -1);
            MyImage     = binarize(MyImage, media(c));
            MyKernel    = simulate_binary_conv(KernelSize,KernelSize);
            SalWI       = conv2(MyImage, MyKernel, 'valid'); %%Filtering
            RelUApp     = max(0,SalWI); %%Apply relu
            
            [threshold, thresholdM] = codificate_window(RelUApp, kernel);
            T = cat(3, T, threshold);
            cont = cont +1;
            cont
        end
    end
    
    
    M = cat(3, M, mode(T, 3));
end

mode(M, 3)


