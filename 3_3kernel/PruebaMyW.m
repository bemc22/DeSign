clear all
close all
m = 3; %%size of the image
n = 3;
pruebas = 500;
T = zeros(m,n,pruebas);
U = zeros(m,n);

for i = 1:pruebas
    
    KernelSize  = 3; %%kernel size
    MyImage     = random_binary_matrix(m,n);
    MyKernel    = random_binary_matrix(KernelSize,KernelSize);
    SalWI       = imfilter(MyImage, MyKernel,'circular','same'); %%Filtering
    RelUApp     = max(0,SalWI); %%Apply relu
    
    [mapping, threshold] = codificate_input_windows_3x3(RelUApp);
    %figure;
    %imagesc(threshold)
    T(:,:,i) = threshold;
    
    igual = 0;
    for k=1:size(U,3)
        if sum(U(:,:,k)~=threshold, 'all') > 0
            igual = igual+1;
        end
    end
    
    if igual==size(U,3)
        U(:,:,size(U,3)+1)=threshold;
    end
    
end

mode(T, 3)
