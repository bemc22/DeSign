function [mapped, thr] = compare2(block, threshold)
thresholds = [0,1,3,5,7,9];
ValRef = block(1,1);

    mapped = zeros(2,2);
    thr = zeros(2, 2);
    found = 1;

    for i=1:2
        for j=1:2

            if ValRef >= thresholds(threshold)
                if found
                    mapped(j,i) = 1;
                end
                thr(j,i) = thresholds(threshold);
                if threshold < 6
                    threshold = threshold+1;
                end
                    
            else
                thr(j, i) = thresholds(threshold);
                found = 0;
                if threshold > 1
                    threshold = threshold-1;
                end
            end
        end
    end
                
             
end