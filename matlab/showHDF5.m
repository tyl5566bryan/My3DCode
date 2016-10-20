%% show results of GAN
clear;close all; clc;

file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration_table_GAN.h5';
data = h5read(file, '/data');
data = permute(data, [5, 4, 3, 2, 1]);
data = squeeze(data);

save_folder = './pictures_GAN/';
if ~exist(save_folder, 'dir'); mkdir(save_folder); end

for i = 1:size(data,1)
    
    sample = data(i, 2:31, 2:31, 2:31);
    sample = squeeze(sample);
    ind1 = find(sample > 0.5); ind2 = find(sample <= 0.5);
    sample(ind1) = 1; sample(ind2) = 0;
    
    figure;
    p = patch(isosurface(squeeze(sample)));
    set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
    daspect([1,1,1])
    view(3); axis off
    camlight
    lighting gouraud

%     figure, plot3D(squeeze(sample)); axis on; grid on;
    print([save_folder num2str(i) '_GAN'], '-dpng');
    close;
    
end

%% show results of AAE
% clear;close all; clc;
% 
% file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration_table_AAE.h5';
% data = h5read(file, '/data');
% data = permute(data, [5, 4, 3, 2, 1]);
% data = squeeze(data);
% data = data(:,2:31,2:31,2:31);
% 
% or_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5';
% or_data = h5read(or_file, '/data');
% or_data = permute(or_data, [4, 3, 2, 1]);
% 
% save_folder = './pictures_AAE/';
% if ~exist(save_folder, 'dir'); mkdir(save_folder); end
% 
% for i = 1:size(or_data,1)
%     
%     sample1 = or_data(i,:,:,:);
%     sample1 = squeeze(sample1);
%     figure;
%     p = patch(isosurface(squeeze(sample1)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
%     
% %     figure, plot3D(squeeze(sample1)); axis on; grid on;
%     print([save_folder num2str(i) '_original'], '-dpng');
%     close;
%     
%     
%     temp = 1;
%     
%     sample2 = data(i,:,:,:);
%     sample2 = squeeze(sample2);
%     ind1 = find(sample2 > 0.1); ind2 = find(sample2 <= 0.1);
%     sample2(ind1) = 1; sample2(ind2) = 0;
%     figure;
%     p = patch(isosurface(squeeze(sample2)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample2)); axis on; grid on;
%     print([save_folder num2str(i) '_recover'], '-dpng');
%     close;
%     
%     temp = 1;
%     
%     sample3 = data(i+64,:,:,:);
%     sample3 = squeeze(sample3);
%     ind3 = find(sample3 > 0.5); ind4 = find(sample3 <= 0.5);
%     sample3(ind3) = 1; sample3(ind4) = 0;
%     
%     figure;
%     p = patch(isosurface(squeeze(sample3)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample3)); axis on; grid on;
%     print([save_folder num2str(i) '_generation'], '-dpng');
%     close;
%     
%     temp = 1;
%     
% end

%% show results of VAE
% clear; close all; clc;
% 
% re_file = '/home/yltian/3D/Code/My3DCode/lua/3Dgeneration_table_VAE.h5';
% or_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5';
% 
% re_data = h5read(re_file, '/data');
% re_data = permute(re_data, [5, 4, 3, 2, 1]);
% re_data = squeeze(re_data);
% re_data = re_data(:,2:31,2:31,2:31);
% 
% or_data = h5read(or_file, '/data');
% or_data = permute(or_data, [4, 3, 2, 1]);
% 
% save_folder = './pictures_VAE/';
% if ~exist(save_folder, 'dir'); mkdir(save_folder); end
% 
% for i = 1:size(or_data,1)
%     sample1 = or_data(i,:,:,:);
%     sample1 = squeeze(sample1);
%     figure;
%     p = patch(isosurface(squeeze(sample1)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample1)); axis on; grid on;
%     print([save_folder num2str(i) '_original'], '-dpng');
%     close;
%     
%     temp = 1;
%     
%     sample2 = re_data(i,:,:,:);
%     sample2 = squeeze(sample2);
%     ind1 = find(sample2 > 0.5); ind2 = find(sample2 <= 0.5);
%     sample2(ind1) = 1; sample2(ind2) = 0;
%     figure;
%     p = patch(isosurface(squeeze(sample2)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample2)); axis on; grid on;
%     print([save_folder num2str(i) '_recover'], '-dpng');
%     close;
%     
%     temp = 1;
%     
%     sample3 = re_data(i+64,:,:,:);
%     sample3 = squeeze(sample3);
%     ind3 = find(sample3 > 0.5); ind4 = find(sample3 <= 0.5);
%     sample3(ind3) = 1; sample3(ind4) = 0;
%     figure;
%     p = patch(isosurface(squeeze(sample3)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample3)); axis on; grid on;
%     print([save_folder num2str(i) '_recover_true'], '-dpng');
%     close;
%     
%     temp = 1;
%     
%     sample4 = re_data(i+128,:,:,:);
%     sample4 = squeeze(sample4);
%     ind5 = find(sample4 > 0.5); ind6 = find(sample4 <= 0.5);
%     sample4(ind5) = 1; sample4(ind6) = 0;
%     figure;
%     p = patch(isosurface(squeeze(sample4)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud
% %     figure, plot3D(squeeze(sample4)); axis on; grid on;
%     print([save_folder num2str(i) '_generation'], '-dpng');
%     close;
%     
%     temp = 1;
% end


%% plot the data
%     p = patch(isosurface(squeeze(sample1)));
%     set(p, 'FaceColor', 'red', 'EdgeColor', 'none');
%     daspect([1,1,1])
%     view(3); axis off
%     camlight
%     lighting gouraud