function shifted = circular_shifting(x, n, right)
% Applies circular shifting to a threshold matrix
thresholds = [0 1 3 5 7 9];
shifted = zeros(size(x));

for k =1:size(x, 3)
    for i=1:size(x,1)
        for j=1:size(x,2)
            pos = x(i, j, k);
            ind = find(thresholds == pos);
            
            if right
                shiftedaux = circshift(thresholds, -n);
            else
                shiftedaux = circshift(thresholds, n);
            end
            shifted(i,j, k) = shiftedaux(ind);
            
        end
    end
    
    
end