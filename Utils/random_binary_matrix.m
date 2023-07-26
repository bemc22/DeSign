function [A] = random_binary_matrix(m,n)
    %%% Creates a random binary matrix
    A = 2*round(rand(m,n)) - 1;
end