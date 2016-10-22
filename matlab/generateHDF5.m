%% exp1
% clc;
% clear;
% 
% source = '/home/yltian/3D/Data/train';
% target = '/home/yltian/3D/Data/train_HDF5_x12';
% 
% if ~isdir(target)
%     mkdir(target)
% end
% 
% folders = dir(source);
% folders = folders(3:end);
% 
% for i = 1:length(folders)
%     folder = [source '/' folders(i).name];
%     files = dir(folder);
%     files = files(3:end);
%     
%     tic
% %     data = zeros(length(files) * 12, 30, 30, 30, 'int8');
%     data = cell(length(files),1);
%     
%     parfor j = 1: length(files)
%         file = [folder '/' files(j).name];
%         volumes = obj2vox_multiview(file, 24, 3, 0);
% %         data((j-1)*12+1:j*12,:,:,:) = volumes; 
% %         data(j,:,:,:) = volume;
%         data{j} = volumes;
%     end
%     data = cat(1, data{:});
%     toc;
%     
%     data = permute(data, [4, 3, 2, 1]);
%     save_name = [target '/' folders(i).name '.h5'];
%     h5create(save_name, '/data', size(data), 'Datatype', 'int8');
%     h5write(save_name, '/data', data);
%     
% end

%% exp2
clc;
clear;

source = '/home/yltian/3D/Data/val';
target = '/home/yltian/3D/Data/val_HDF5';

if ~isdir(target)
    mkdir(target)
end

opt.num = 64;
opt.type = '03001627';


folder = [source '/' opt.type];
files = dir(folder);
files = files(3 : 3 + opt.num - 1);

data = zeros(opt.num, 30, 30, 30, 'int8');

for i = 1:length(files)
    file = [folder '/' files(i).name];
    volume = obj2vox(file, 24, 3, 0);
    data(i,:,:,:) = volume;
end

data = permute(data, [4, 3, 2, 1]);

save_name = [target '/' opt.type '.h5'];
h5create(save_name, '/data', size(data), 'Datatype', 'int8');
h5write(save_name, '/data', data);

