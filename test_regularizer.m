addpath("regularizer\");


I = imread("cameraman.tif");
I = double(I) / 255.0;


Inoisy = I + randn(size(I))*10/255;
Inoisy=min(max(Inoisy,0),1);


tv(Inoisy)
mse(I,  Inoisy)
psnr(I, Inoisy)
corr(I, Inoisy)
wpsnr(I, Inoisy)






