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
   lr_ae = 0.002,
   lr_adv = 0.01,
   beta1_ae = 0.9,
   beta1_adv = 0.1,
   learningRateDecay = 0.0002,
   nz = 25,
   nd = 500,
   nh = 500,
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
--torch.setDevice(opt.gpu + 2)

-- create data sampler
local data_sampler = VoxelBatchSampler(opt.hdf5_files)

-----------------------------------------------------
local inputs = torch.Tensor(opt.batch_size, 1, opt.cube_length, opt.cube_length, opt.cube_length):fill(0)
local noise = torch.Tensor(2 * opt.batch_size, opt.nz)
local label = torch.Tensor(2 * opt.batch_size)

local err_rec, err_D, err_G
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
   elseif name:find('Linear') then
      m.weight:normal(0.0, 0.01)
   end
end

-------------------------------------------------------------------
local VolumetricConv = nn.VolumetricConvolution
local VolumetricFullConv = nn.VolumetricFullConvolution
local VolumetricBN = nn.VolumetricBatchNormalization
local View = nn.View
local Linear = nn.Linear
local BN = nn.BatchNormalization

local nef = opt.nef
local ndf = opt.ndf
local nz = opt.nz
local nh = opt.nh
local nd = opt.nd
local nc = opt.nc

local encoder = nn.Sequential()
encoder:add(VolumetricConv(nc, nef, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef, nef * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 2)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(nef * 2, nef * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
encoder:add(VolumetricBN(nef * 4)):add(nn.LeakyReLU(0.2, true))
encoder:add(VolumetricConv(ndf * 4, nh, 4, 4, 4))
encoder:add(VolumetricBN(nh))
encoder:add(View(nh)):add(nn.ReLU(true))
encoder:add(Linear(nh, nz))

local decoder = nn.Sequential()
decoder:add(Linear(nz, nh)):add(View(nh, 1, 1, 1))
decoder:add(VolumetricBN(nh)):add(nn.LeakyReLU(0.2, true))
decoder:add(VolumetricFullConv(nh, ndf * 4, 4, 4, 4))
decoder:add(VolumetricBN(ndf * 4)):add(nn.LeakyReLU(0.2, true))
decoder:add(VolumetricFullConv(ndf * 4, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(VolumetricBN(ndf * 2)):add(nn.LeakyReLU(0.2, true))
decoder:add(VolumetricFullConv(ndf * 2, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(VolumetricBN(ndf)):add(nn.LeakyReLU(0.2, true))
decoder:add(VolumetricFullConv(ndf, nc, 4, 4, 4, 2, 2, 2, 1, 1, 1))
decoder:add(nn.Sigmoid())
decoder:apply(weights_init)

local autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)

local discriminator = nn.Sequential()
discriminator:add(Linear(nz, nd))
discriminator:add(BN(nd)):add(nn.ReLU(true))
discriminator:add(Linear(nd, nd))
discriminator:add(BN(nd)):add(nn.ReLU(true))
discriminator:add(Linear(nd, 1))
discriminator:add(nn.Sigmoid())
discriminator:apply(weights_init)

local rec_criterion = nn.BCECriterion()

local adv_criterion = nn.BCECriterion()

-----------------------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   inputs = inputs:cuda()
   noise = noise:cuda()
   label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(encoder, cudnn) 
      cudnn.convert(decoder, cudnn)
      cudnn.convert(autoencoder, cudnn)
      cudnn.convert(discriminator, cudnn)
   end
   encoder:cuda(); decoder:cuda(); autoencoder:cuda(); 
   discriminator:cuda(); rec_criterion:cuda(); adv_criterion:cuda()
end

optimState = {
  learningRate = opt.lr_ae,
  beta1 = opt.beta1_ae,
  learningRateDecay = opt.learningRateDecay,
}
optimState_D = {
  learningRate = opt.lr_adv,
  beta1 = opt.beta1_adv,
  learningRateDecay = opt.learningRateDecay,
}

local parameters, gradParameters = autoencoder:getParameters()
local parameters_D, gradParameters_D = discriminator:getParameters()

local f1 = function(x)
  if parameters ~= x then
    parameters:copy(x)
  end
  
  gradParameters:zero()
  gradParameters_D:zero()
  
  data_tm:reset(); data_tm:resume()
  local real = data_sampler:sampleBalance(opt.batch_size)
  data_tm:stop()
  real = torch.reshape(real, torch.LongStorage{opt.batch_size, 1, 30, 30, 30})
  inputs:fill(0)
  inputs[{{},{},{2,31},{2,31},{2,31}}]:copy(real)
  
  -- autoencoder reconstruction loss
  local inputs_rec = autoencoder:forward(inputs)
  err_rec = rec_criterion:forward(inputs_rec, inputs)
  local grad_rec = rec_criterion:backward(inputs_rec, inputs)
  autoencoder:backward(inputs, grad_rec)
  
  -- train discriminator with prior distribution
  noise[{{1,opt.batch_size},{}}]:normal(0, 1)
  label[{{1,opt.batch_size}}]:fill(1)
  noise[{{1 + opt.batch_size, 2 * opt.batch_size},{}}]:copy(encoder.output)
  label[{{1 + opt.batch_size, 2 * opt.batch_size}}]:fill(0)
  local pred = discriminator:forward(noise)
  local loss = adv_criterion:forward(pred, label)
  local gradrealLoss = adv_criterion:backward(pred, label)
  discriminator:backward(noise, gradrealLoss)
  
  err_D = loss
  
  -- train encoder (generator) to play the minimax game
  --label:fill(1)
  local pred2 = discriminator:forward(encoder.output)
  err_G = adv_criterion:forward(pred2, label[{{1,opt.batch_size}}])
  local gradminimaxloss = adv_criterion:backward(pred2, label[{{1,opt.batch_size}}])
  local gradminimax = discriminator:updateGradInput(encoder.output, gradminimaxloss)
  encoder:backward(inputs, gradminimax)
  
  return err_rec, gradParameters
end

local f2 = function(x)
  if parameters_D ~= x  then
    parameters_D:copy(x)
  end
  
  return  err_D, gradParameters_D
end
---------------------------------------------------------------------

for epoch = 1, opt.nepoch do
  epoch_tm:reset()
  for i = 1, math.min(data_sampler:size(), opt.ntrain), opt.batch_size do
  --for i = 1, 40, opt.batch_size do
    batch_tm:reset()
    
    -- Update 3D Autoencoder
    optim.adam(f1, parameters, optimState)
    -- Update Discriminator
    optim.adam(f2, parameters_D, optimState_D)
    
    -- logging
      if ((i-1) / opt.batch_size) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_REC: %.4f  Err_G: %.4f Err_D: %4f'):format(
                 epoch, ((i-1) / opt.batch_size),
                 math.floor(math.min(data_sampler:size(), opt.ntrain) / opt.batch_size),
                 batch_tm:time().real, data_tm:time().real,
                 err_rec and err_rec or -1, err_G and err_G or -1, err_D and err_D or -1))
      end
  end
  
  paths.mkdir('checkpoints')
  parameters, gradParameters = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_AAE.t7', autoencoder:clearState())
  
  local parameters_E, gradParameters_E = encoder:getParameters()
  parameters_E, gradParameters_E = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_encoder.t7', encoder:clearState())
  
  local parameters_D, gradParameters_D = decoder:getParameters()
  parameters_D, gradParameters_D = nil, nil -- nil them to avoid spiking memory
  torch.save('checkpoints/' .. opt.name .. '_' .. epoch .. '_decoder.t7', decoder:clearState())
  
  parameters, gradParameters = autoencoder:getParameters() -- reflatten the params and get them
  print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.nepoch, epoch_tm:time().real))
  
end

