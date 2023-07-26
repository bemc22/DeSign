function comp = complement(x)
% Finds the complement of a threshold matrix

comp = zeros(size(x));
for i=1:size(x,1)
    for j=1:size(x,2)
        
        switch x(i,j)
            case 0
                comp(i,j) = 9;
            case 1
                comp(i,j) = 7;
            case 3
                comp(i,j) = 5;
            case 5
                comp(i,j) = 3;
            case 7
                comp(i,j) = 1;
            case 9
                comp(i,j) = 0;
        end
    end
    
    
end
end
