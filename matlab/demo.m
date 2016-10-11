% Demo RUN ME
% Requirement: voxnet_feature.m, conv3d.m, maxpooling3d.m

% clear;close all; clc;

% mex polygon2voxel_double.c -v

% load obj and convert it to occupancy grid in 1x30x30x30
% load('example_input.mat');

% load obj file and convert it to binary volume
volume = obj2vox('piano_0001.obj', 24, 3, 0); % up direction is +z
% volume = obj2vox('model_normalized.obj', 24, 3, 0); % up direction is +z
% volume = obj2vox('/home/yltian/3D/Data/val/02801938/model_010402.obj', 24, 3, 0);
input = reshape(volume, 1, 30, 30, 30);
input = double(input);
figure, plot3D(squeeze(input)); axis on; grid on;

% % load voxnet parameters
% load('voxnet_params_ours.mat')
% load('subvolumesup_params_ours.mat')
% 
% % extract feature
% tic;
% feat = voxnet_feature(input, voxnet_params);
% toc;
% 
% tic;
% feat = subvolumesup_feature(input, subvolumesup_params);
% toc;
% 
% % visualize the object
% fprintf('Press any key to visualize the object..\n');
% pause();
% figure, plot3D(squeeze(input)); axis on; grid on;