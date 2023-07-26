clear all
close all
clc
pruebas = 100;
T = zeros(3,3,pruebas);


%%Parameters
m = 3; %%size of the image
n = 3;
KernelSize  = 3; %%kernel size
U = zeros(m,n);
for i=1:pruebas
MyImage     = random_binary_matrix(m,n);
MyKernel    = random_binary_matrix(KernelSize,KernelSize);
SalWI       = imfilter(MyImage, MyKernel,'circular','same'); %%Filtering
RelUApp     = max(0,SalWI); %%Apply relu


%%Mapping approximation
MyMapping   = zeros(size(RelUApp));
MyTHR_m     = zeros(size(RelUApp));
thrV        = [0 1 3 5 7 9]; %%Possibles thresholdings

%RefLeft     = 1;
%RefTop      = 1;

CountInd    = [];

%IndLin      = sub2ind([m,n],RefLeft,RefTop);
IndLin      = 0;
while (length(CountInd) < m*n)
    IndLin      = IndLin + 1;
    indCamb     = 0;
    indThr      = 2;
    [RefLeft, RefTop] = ind2sub([m,n],IndLin);
    
    %%Saber indices realmente disponibles 
    [A,B] = meshgrid(RefLeft:min(RefLeft+1,m),RefTop:min(RefTop+1,n));
    aux   = sort(sub2ind([m,n],A(:),B(:)),'ascend');
    aux   = aux(find(~ismember(aux,CountInd)));
    
    ValRef      = RelUApp(RefLeft,RefTop);
    AnsQ        = ValRef >= thrV(indThr);
    ss          = zeros(1,length(aux));
    thr_tmp     = zeros(1,length(aux));
    ss(1)       = AnsQ;
    thr_tmp(1)  = indThr;
    MyCuenta    = 1;
    if AnsQ == 1 && indThr < size(thrV,2)
        indThr = indThr + 1;
    end
    if AnsQ == 0 && indThr > 1
        indThr = indThr - 1;
    end
    if (length(aux)==1)
        indCamb = 1;
        MyMapping(aux(1:MyCuenta)) = ss(1:MyCuenta);
        MyTHR_m(aux(1:MyCuenta)) = thr_tmp(1:MyCuenta);
        CountInd   = [CountInd; aux];
    end
    while indCamb == 0 && MyCuenta < length(aux)
        MyCuenta = MyCuenta + 1;
        AnsQ     = ValRef >= thrV(indThr);
        ss(MyCuenta) = AnsQ;
        thr_tmp(MyCuenta)  = indThr;
        if AnsQ == 1 && indThr < size(thrV,2)
            indThr = indThr + 1;
        end
        if AnsQ == 0 && indThr > 1
            indThr = indThr - 1;
        end
        if (ss(MyCuenta) ~= ss(MyCuenta-1) || MyCuenta == length(aux))
            indCamb = 1;
            MyMapping(aux(1:MyCuenta)) = ss(1:MyCuenta);
            MyTHR_m(aux(1:MyCuenta)) = thr_tmp(1:MyCuenta);
            CountInd   = [CountInd; aux];
        end
    end
end
T(:,:,i)=MyTHR_m;

%figure;
%imagesc(MyTHR_m)

igual = 0;
for k=1:size(U,3)
    if sum(U(:,:,k)~=MyTHR_m, 'all') > 0
        igual = igual+1;  
    end   
end

if igual==size(U,3)
      U(:,:,size(U,3)+1)=MyTHR_m;     
end


end

mode(T,3)
