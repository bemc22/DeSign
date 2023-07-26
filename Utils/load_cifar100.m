function [CIFAR100] = load_cifar100()
% Loads Cifar100 Dataset
train = load('cifar-100-matlab/train.mat'); 
test = load('cifar-100-matlab/test.mat'); 

train = double(train.data);
test = double(test.data);

CIFAR100 = cat(1, train, test)./255;
CIFAR100 = reshape(CIFAR100, [size(CIFAR100, 1), 32, 32, 3]);

end