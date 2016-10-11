require 'torch'
require 'image'
require 'hdf5'

local Dataloader = torch.class('VoxelDataLoader')

function Dataloader:__init(file)
  self.file = file
  
  print('DataLoader loading h5 file: ', file)
  self.h5_file = hdf5.open(file, 'r')
  self.data = self.h5_file:read('/data'):all()
  
  local voxel_sizes = self.h5_file:read('/data'):dataspaceSize()
  assert(#voxel_sizes == 4, 'voxels should be a 4-D tensor')
  assert(voxel_sizes[2] == voxel_sizes[3] and voxel_sizes[3] == voxel_sizes[4], 'voxels should be cubic')
  
  self.num_voxels = voxel_sizes[1]
  self.cube_length = voxel_sizes[2]
  
  self.iterator = 1
  self.shuffle = torch.linspace(1, self.num_voxels, self.num_voxels)
  self.default_batch_size = 64
end

function Dataloader:size()
  return self.num_voxels
end

function Dataloader:ShuffleData()
  self.shuffle = torch.randperm(self.num_voxels)
end

function Dataloader:sample(n)
  local batch_size = n or self.default_batch_size
  local len = self.cube_length
  local res = torch.ByteTensor(batch_size, len, len, len)
  for i = 1, batch_size do
    res[i] = self.data[self.shuffle[self.iterator]]
    self.iterator = self.iterator + 1
    if self.iterator > self.num_voxels then
      self.iterator = 1
      self.shuffle = torch.randperm(self.num_voxels)
    end
  end
  --res = res - 0.5
  return res
end

  