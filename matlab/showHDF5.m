%% show generating results of GAN
clear;close all; clc;

file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration_table_iter25.h5';
data = h5read(file, '/data');
data = permute(data, [5, 4, 3, 2, 1]);
data = squeeze(data);

for i = 1:size(data,1)
    
    sample = data(i, 2:31, 2:31, 2:31);
    sample = squeeze(sample);
    ind1 = find(sample > 0.5); ind2 = find(sample <= 0.5);
    sample(ind1) = 1; sample(ind2) = 0;
    sample = double(sample);

    figure, plot3D(squeeze(sample)); axis on; grid on;
end

%% show reconstruction results of VAE
clear; close all; clc;

re_file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration_VAE.h5';
or_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5';

re_data = h5read(re_file, '/data');
re_data = permute(re_data, [5, 4, 3, 2, 1]);
re_data = squeeze(re_data);
re_data = re_data(:,2:31,2:31,2:31);

or_data = h5read(or_file, '/data');
or_data = permute(or_data, [4, 3, 2, 1]);

for i = 1:size(or_data,1)
    sample1 = or_data(i,:,:,:);
    sample1 = squeeze(sample1);
    figure, plot3D(squeeze(sample1)); axis on; grid on;
    print(['./pictures/' num2str(i) '_original'], '-dpng');
    close;
    
    temp = 1;
    
    sample2 = re_data(i,:,:,:);
    sample2 = squeeze(sample2);
    ind1 = find(sample2 > 0.5); ind2 = find(sample2 <= 0.5);
    sample2(ind1) = 1; sample2(ind2) = 0;
    figure, plot3D(squeeze(sample2)); axis on; grid on;
    print(['./pictures/' num2str(i) '_recover'], '-dpng');
    close;
    
    temp = 1;
    
    sample3 = re_data(i+128,:,:,:);
    sample3 = squeeze(sample3);
    ind3 = find(sample3 > 0.5); ind4 = find(sample3 <= 0.5);
    sample3(ind3) = 1; sample3(ind4) = 0;
    figure, plot3D(squeeze(sample3)); axis on; grid on;
    print(['./pictures/' num2str(i) '_recover_true'], '-dpng');
    close;
    
    temp = 1;
end