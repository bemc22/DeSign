function [c, h] = create_histogram(matrix)

[n,m,p] = size(matrix);
a = reshape(matrix, n, [], 1);
b = reshape(a(:), n*m, [])';
[c, ia, ic] = unique(b, 'rows', 'first');
c = c';
c = reshape(c, n, m, []);

ic = categorical(ic);
figure;
h = histogram(ic, 'Normalization','pdf');
h.DisplayOrder = 'descend';
xlabel('Indice de threshold');
ylabel('Probabilidad');

end