function volumes = obj2vox_multiview(obj_filename, volume_size, pad_size, visu)
% OBJ2VOX, convert an obj model to binary volume with multi-views

FV = obj_loader(obj_filename);
% FV = rotateTest(FV);
size = volume_size + 2 * pad_size;
volumes = zeros(12, size, size, size, 'int8');

for i = 0:11
    theta = i * pi / 6;
    fv_rotate = rotateTest(FV, theta);
    volume = polygon2voxel(fv_rotate, [volume_size, volume_size, volume_size], 'auto', 1, 0);
    volume = padarray(volume, [pad_size, pad_size, pad_size]);
    volume = int8(volume);
    volumes(i+1,:,:,:) = volume;
end

end
