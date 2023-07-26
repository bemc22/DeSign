function [T, thresholds] = codificate_window(matrix, kernel)
rows = size(matrix, 1);
columns = size(matrix, 2);
mapping = zeros(rows, columns);
thresholds = zeros(rows, columns);
threshold = 2;

T = [];

for i=1:columns/kernel
    for j=1:rows/kernel
        block = matrix((kernel*j-(kernel-1)):(kernel*j-(kernel-1))+(kernel-1), (kernel*i-(kernel-1)):(kernel*i-(kernel-1))+(kernel-1));
        [mapped, thr] = compare_window_threshold(block, threshold, kernel);
        if sum(thr(:)) < 1
            thr = [];
        else
            T = cat(3, T, thr);
        end
        
        thresholds((kernel*j-(kernel-1)):(kernel*j-(kernel-1))+(kernel-1), (kernel*i-(kernel-1)):(kernel*i-(kernel-1))+(kernel-1)) = thr;
        
    end
end
end