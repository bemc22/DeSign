T1 = load('2x2threshold.mat');
T2 = load('2x2threshold2.mat');
T1 = T1.T;
T2 = T2.T;
A = cat(3, T1, T2);
[c, h] = create_histogram(A);

T1 = load('3x3threshold.mat');
T1 = T1.T;
[c2, h2] = create_histogram(T1);