function quantized = apply_halfwave(x)
%Applies halfwave to integer threshold matrix
quantized = zeros(size(x));
for k = 1:size(x, 3)
    for i=1:size(x, 1)
        for j=1:size(x,2)
            switch(x(i,j,k))
                case 0
                    quantized(i,j,k) = 0.5159260799279024;
                case 1
                    quantized(i,j,k) = 1.0426740122251235;
                case 3
                    quantized(i,j,k) = 1.5971041732997648;
                case 5
                    quantized(i,j,k) = 2.2059878729752223;
                case 7
                    quantized(i,j,k) = 2.9180711418567356;
                case 9
                    quantized(i,j,k) = 3.8667480105918006;
            end
        end
    end
end