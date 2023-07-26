function reluapp = simulate_binary_conv(ImageSize, KernelSize)
    %%% Simulates a binary convolution and applies ReLU function to the result.

    MyImage     = random_binary_matrix(ImageSize,ImageSize);
    MyKernel    = random_binary_matrix(KernelSize,KernelSize);
    SalWI       = conv2(MyImage, MyKernel, 'valid'); %%Filtering
    reluapp     = max(0,SalWI); %%Apply relu

end