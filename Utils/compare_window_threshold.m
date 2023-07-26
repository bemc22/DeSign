function [mapped, thr] = compare_window_threshold(block, threshold, kernel)
%Compares a window with the threshold set applying the adaptive strategy
%designed
thresholds = [0, 1, 3, 5, 7, 9];

mapped = zeros(kernel, kernel);
thr = zeros(kernel, kernel);

for i=1:kernel
    for j=1:kernel
        
        if block(j,i) >= thresholds(threshold)
            mapped(j,i) = 1;
            thr(j,i) = thresholds(threshold);
            if threshold < 6
                threshold = threshold+1;
            end
            
        else
            thr(j, i) = thresholds(threshold);
            if threshold > 1
                threshold = threshold-1;
            end
        end
    end
end
end