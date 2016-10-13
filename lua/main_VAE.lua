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
   nef = 64,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   nepoch = 25,
   G_k = 5;
   D_k = 1;
   ntrain = math.huge,
   --ntrain = 128,
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

local err_KLD, err_Rec, err_bound
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

local nef = opt.nef
local ndf = opt.ndf
local nz = opt.nz
local nc = opt.nc
local nh = opt.nh

local encoder = nn.Sequential()
encoder:add(VolumetricConv(nc, nef, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef, nef * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 2)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef * 2, nef * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 4)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(ndf * 4, nh, 4, 4, 4))
encoder:add(View(nh)):add(nn.ReLU(true))
local mean_logvar = nn.ConcatTable()
mean_logvar:add(Linear(nh, nz))
mean_logvar:add(Linear(nh, nz))
encoder:add(mean_logvar)
encoder:apply(weights_init)

local decoder = nn.Sequential()
decoder:add(Linear(nz, nh)):add(nn.ReLU(true))
decoder:add(View(nh, 1, 1, 1))
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

optimState = {
  learningRate = 0.001
}

local parameters, gradParameters = model:getParameters()

local fx = function(x)
  gradParameters:zero()
  --model:zeroGradParameters()
  
  data_tm:reset(); data_tm:resume()
  local real = data_sampler:sampleBalance(opt.batch_size)
  data_tm:stop()
  real = torch.reshape(real, torch.LongStorage{opt.batch_size, 1, 30, 30, 30})
  inputs:fill(0)
  inputs[{{},{},{2,31},{2,31},{2,31}}]:copy(real)
  
  local reconstruction, mean, log_var
  reconstruction, mean, log_var = unpack(model:forward(inputs))
  
  -- reconstruction loss
  err_Rec = criterion:forward(reconstruction, inputs)
  local df_dw = criterion:backward(reconstruction, inputs)
  
  -- KLD loss
  local KLD = 1 + log_var - torch.pow(mean, 2) - torch.exp(log_var)
  err_KLD = -0.5 * torch.sum(KLD)
  local dKLD_dmu = mean:clone()
  local dKLD_dlog_var = torch.exp(log_var):mul(-1):add(1):mul(-0.5)
  
  -- lower bound
  err_bound = err_Rec + err_KLD
  
  error_grads = {df_dw, dKLD_dmu, dKLD_dlog_var}
  model:backward(inputs, error_grads)
  
  return err_bound, gradParameters
end
---------------------------------------------------------------------

for epoch = 1, opt.nepoch do
  epoch_tm:reset()
  for i = 1, math.min(data_sampler:size(), opt.ntrain), opt.batch_size do
  --for i = 1, 40, opt.batch_size do
    batch_tm:reset()
    
    -- Update 3D convolutional VAE
    optim.adam(fx, parameters, optimState)
    
    -- logging
      if ((i-1) / opt.batch_size) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_REC: %.4f  Err_KLD: %.4f Err_BND: %4f'):format(
                 epoch, ((i-1) / opt.batch_size),
                 math.floor(math.min(data_sampler:size(), opt.ntrain) / opt.batch_size),
                 batch_tm:time().real, data_tm:time().real,
                 err_Rec and err_Rec or -1, err_KLD and err_KLD or -1, err_bound and err_bound or -1))
      end
  end
  
  paths.mkdir('checkpoints')
  parameters, gradParameters = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_VAE.t7', model:clearState())
  
  local parameters_E, gradParameters_E = encoder:getParameters()
  parameters_E, gradParameters_E = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_Encoder.t7', encoder:clearState())
  
  local parameters_D, gradParameters_D = decoder:getParameters()
  parameters_D, gradParameters_D = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_Decoder.t7', decoder:clearState())
  
  parameters, gradParameters = model:getParameters() -- reflatten the params and get them
  print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.nepoch, epoch_tm:time().real))
  
end

