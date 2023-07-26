function [T] = codificate_window(matrix)
    rows = size(matrix, 1);
    columns = size(matrix, 2);
    mapping = zeros(rows, columns);
    thresholds = zeros(rows, columns);
    threshold = 2;
    
    T = [];

    for i=1:columns/2
        for j=1:rows/2
            block = matrix((2*j-1):(2*j-1)+1, (2*i-1):(2*i-1)+1);
            [mapped, thr] = compare(block, threshold);
            if sum(thr(:)) < 1
                thr = [];
            else
                T = cat(3, T, thr);
            end
        end
    end
end