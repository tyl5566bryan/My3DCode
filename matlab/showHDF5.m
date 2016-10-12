% show generating results

% clear;close all; clc;

file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration.h5';
data = h5read(file, '/data');
data = permute(data, [5, 4, 3, 2, 1]);
data = squeeze(data);

i = 2;
sample = data(i, 2:31, 2:31, 2:31);
sample = squeeze(sample);
ind1 = find(sample > 0.99); ind2 = find(sample <= 0.99);
sample(ind1) = 1; sample(ind2) = 0;
sample = double(sample);

figure, plot3D(squeeze(sample)); axis on; grid on;


