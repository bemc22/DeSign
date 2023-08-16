function [binarized] = binarize_pos(matrix, threshold)
%Binarize matrix with a defined threshold
matrix(matrix>=threshold) = 1;
matrix(matrix<threshold) = 0;

binarized = matrix;

end