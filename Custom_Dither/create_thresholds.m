function [thresholds] = create_thresholds(dimension)
    
    permutations = combinator(6, dimension, 'p', 'r');
    permutations = 2.*permutations -1;
    permutations(permutations == 11) = 0;
    total_p = size(permutations, 1);
    thresholds = reshape(permutations', [sqrt(dimension), sqrt(dimension), total_p]);

end

