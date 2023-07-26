clear all
close all
T =[];
for h=1:5
    [images, original] = load_cifar10(h);
    %T = zeros(2,2,8410000);
    %U = zeros(2,2);
    h
    
    for i = 1:size(images,1)
        
        KernelSize  = 3; %%kernel size
        MyImage     = squeeze(images(i,:,:));
        MyKernel    = random_binary_matrix(KernelSize,KernelSize);
        SalWI       = conv2(MyImage, MyKernel, 'valid'); %%Filtering
        RelUApp     = max(0,SalWI); %%Apply relu
        
        [threshold] = codificate_input_windows_2x2(RelUApp);
        T = cat(3, T, threshold);
        %Printing image structure
        %figure;
        %subplot(1,2,1)
        %imagesc(squeeze(original(i,:,:)))
        %colorbar()
        %colormap gray
        %subplot(1,2,2)
        %imagesc(threshold)
        %colorbar()
        %colormap gray
        %pause(1)
        %close all     
    end
end

mode(T, 3)
