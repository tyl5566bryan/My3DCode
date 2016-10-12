require 'torch'
require 'nn'
require 'optim'
require 'nngraph'

require 'batchSampler'
require 'Sampler'

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
     '/home/yltian/3D/Data/train_HDF5_x12/04379243.h5',
     
     --[[
     '/home/yltian/3D/Data/train_HDF5/02691156.h5',
     '/home/yltian/3D/Data/train_HDF5/02828884.h5',
     '/home/yltian/3D/Data/train_HDF5/02933112.h5',
     '/home/yltian/3D/Data/train_HDF5/02958343.h5',
     '/home/yltian/3D/Data/train_HDF5/03001627.h5',
     '/home/yltian/3D/Data/train_HDF5/04256520.h5',
     '/home/yltian/3D/Data/train_HDF5/04379243.h5',
     --]]
   },
   gpu = 1,
   batch_size = 64,
   cube_length = 32,
   lr = 0.001, 
   beta1 = 0.5, 
   nh = 400,
   nz = 100,
   nc = 1,
   nef = 32,               -- #  of gen filters in first conv layer
   ndf = 32,               -- #  of discrim filters in first conv layer
   nepoch = 25,
   G_k = 5;
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
local data_sampler = VoxelBatchSampler(opt.hdf5_files)

-----------------------------------------------------
local inputs = torch.Tensor(opt.batch_size, 1, opt.cube_length, opt.cube_length, opt.cube_length):fill(0)

local err_KLD, err_REC
local epoch_tm = torch.Timer()
local batch_tm = torch.Timer()
local data_tm  = torch.Timer()

-----------------------------------------------------
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      --m:noBias()
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-------------------------------------------------------------------
local VolumetricConv = nn.VolumetricConvolution
local VolumetricFullConv = nn.VolumetricFullConvolution
local VolumetricBN = nn.VolumetricBatchNormalization
local View = nn.View
local Linear = nn.Linear
local Reshape = nn.Reshape

local nef = opt.nef
local ndf = opt.ndf
local nz = opt.nz
local nc = opt.nc
local nh = opt.nh
local batch = opt.batch_size

local encoder = nn.Sequential()
encoder:add(VolumetricConv(nc, nef, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef, nef * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 2)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef * 2, nef * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 4)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(ndf * 4, nh, 4, 4, 4))
encoder:add(Reshape(batch, nh)):add(nn.ReLU(true))
local mean_logvar = nn.ConcatTable()
mean_logvar:add(Linear(nh, nz))
mean_logvar:add(Linear(nh, nz))
encoder:add(mean_logvar)
encoder:apply(weights_init)

local decoder = nn.Sequential()
decoder:add(Linear(nz, nh)):add(nn.ReLU(true))
decoder:add(Reshape(batch, nh, 1, 1, 1))
decoder:add(VolumetricFullConv(nh, ndf * 4, 4, 4, 4))
decoder:add(VolumetricBN(ndf * 4)):add(nn.ReLU(true))
decoder:add(VolumetricFullConv(ndf * 4, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(VolumetricBN(ndf * 2)):add(nn.ReLU(true))
decoder:add(VolumetricFullConv(ndf * 2, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(VolumetricBN(ndf)):add(nn.ReLU(true))
decoder:add(VolumetricFullConv(ndf, nc, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(nn.Sigmoid())
decoder:apply(weights_init)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})
local reconstruction = decoder(z)
local model = nn.gModule({input},{reconstruction, mean, log_var})

local criterion = nn.BCECriterion()
criterion.sizeAverage = false

-----------------------------------------------------------------
optimState = {
  learningRate = 0.001
}

if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   inputs = inputs:cuda(); 

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(encoder, cudnn); cudnn.convert(decoder, cudnn); cudnn.convert(model, cudnn)
   end
   encoder:cuda(); decoder:cuda(); model:cuda(); criterion:cuda()
end

local parameters, gradParameters = model:getParameters()

local fx = function(x)
  gradParametersD:zero()
  --model:zeroGradParameters()
  
  
end
---------------------------------------------------------------------



local KLDloss
local gradKLDloss

local debug

