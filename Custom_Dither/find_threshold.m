function [idx] = find_threshold(threshold, thresholds)
%Returns the index of an specific threshold inside all possible thresholds
found = 0;
i = 1;

while not(found)
    if threshold == thresholds(:,:,i)
        idx = i;
        found = 1;
    end
    
    i = i+1;
end

end