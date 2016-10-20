require 'torch'
require 'nn'
require 'optim'
require 'batchSampler'

-- set torch environment
opt = {
   hdf5_files = {
     --[['/home/yltian/3D/Data/train_HDF5_x12/02691156.h5',
     '/home/yltian/3D/Data/train_HDF5_x12/02828884.h5',
     '/home/yltian/3D/Data/train_HDF5_x12/02933112.h5',
     '/home/yltian/3D/Data/train_HDF5_x12/02958343.h5',
     '/home/yltian/3D/Data/train_HDF5_x12/03001627.h5',
     '/home/yltian/3D/Data/train_HDF5_x12/04256520.h5',
     --]]
     --'/home/yltian/3D/Data/train_HDF5_x12/04379243.h5',
     
     '/home/yltian/3D/Data/train_HDF5/02691156.h5',
     '/home/yltian/3D/Data/train_HDF5/02828884.h5',
     '/home/yltian/3D/Data/train_HDF5/02933112.h5',
     '/home/yltian/3D/Data/train_HDF5/02958343.h5',
     '/home/yltian/3D/Data/train_HDF5/03001627.h5',
     '/home/yltian/3D/Data/train_HDF5/04256520.h5',
     '/home/yltian/3D/Data/train_HDF5/04379243.h5',
   },
   gpu = 1,
   batch_size = 64,
   cube_length = 32,
   lr = 0.0002, 
   beta1 = 0.5, 
   nz = 100,
   nc = 1,
   ngf = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nepoch = 25,
   G_k = 3;
   D_k = 1;
   ntrain = math.huge,
   name = '3DShape',
}
opt.manualSeed = torch.random(1, 10000)
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data sampler
local sampler = VoxelBatchSampler(opt.hdf5_files)

-----------------------------------------------------
local input = torch.Tensor(opt.batch_size, 1, opt.cube_length, opt.cube_length, opt.cube_length):fill(0)
local noise = torch.Tensor(opt.batch_size, opt.nz, 1, 1, 1)
local label = torch.Tensor(opt.batch_size)

local epoch_tm = torch.Timer()
local iter_tm = torch.Timer()
local data_tm = torch.Timer()

local real_label = 1
local fake_label = 0

-----------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      --m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   elseif name:find('Linear') then
      if m.weight then m.weight:normal(0.0, 0.01)
      if m.bias then m.bias:fill(0) end
   end
end

local VolumetricConv = nn.VolumetricConvolution
local VolumetricFullConv = nn.VolumetricFullConvolution
local VolumetricBN = nn.VolumetricBatchNormalization

local ngf = opt.ngf
local ndf = opt.ndf
local nz = opt.nz
local nc = opt.nc

local netG = nn.Sequential()

netG:add(VolumetricFullConv(nz, ngf * 4, 4, 4, 4))
netG:add(VolumetricBN(ngf * 4)):add(nn.ReLU(true))
netG:add(VolumetricFullConv(ngf * 4, ngf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netG:add(VolumetricBN(ngf * 2)):add(nn.ReLU(true))
netG:add(VolumetricFullConv(ngf * 2, ngf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netG:add(VolumetricBN(ngf)):add(nn.ReLU(true))
netG:add(VolumetricFullConv(ngf, nc, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netG:add(nn.Sigmoid())
--netG:add(nn.Tanh())

netG:apply(weights_init)

local netD = nn.Sequential()
netD:add(VolumetricConv(nc, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
netD:add(VolumetricConv(ndf, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netD:add(VolumetricBN(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netD:add(VolumetricConv(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netD:add(VolumetricBN(ndf * 4)):add(nn.LeakyReLU(0.2, true))
netD:add(VolumetricConv(ndf * 4, 1, 4, 4, 4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(4))

netD:apply(weights_init)

local criterion = nn.BCECriterion()