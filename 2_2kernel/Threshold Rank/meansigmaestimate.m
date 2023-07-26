pruebas = 400;

medias = zeros(pruebas, 1);
desviaciones = zeros(pruebas, 1);

for k=1:pruebas
    
    matrix = zeros(50000*30*30,1);
    
    for h=1:5
        [images, original] = load_cifar10(h);
        aux = zeros(10000*30*30,1);
        for j=1:size(images , 1)
            KernelSize  = 3; %%kernel size
            MyImage     = squeeze(images(j,:,:));
            MyKernel    = BinN(KernelSize,KernelSize);
            SalWI       = conv2(MyImage, MyKernel, 'valid'); %%Filtering
            aux((j-1)*30*30+1:(j-1)*30*30+30*30) = SalWI(:);
        end
        
        matrix((h-1)*30*30*10000+1:(h-1)*30*30*10000+10000*30*30) = aux;
        
    end
    
medias(k,1) = mean(matrix);
desviaciones(k,1) = std(matrix);

end

mean(desviaciones)
mean(medias)

