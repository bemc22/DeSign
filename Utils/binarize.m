function [binarized] = binarize(matrix, threshold)
%Binarize matrix with a defined threshold
matrix(matrix>threshold) = 1;
matrix(matrix<=threshold) = -1;

binarized = matrix;

end