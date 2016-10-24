require 'torch'
require 'nn'
require 'optim'
require 'batchSampler'
require 'utils/VolumetricProjection'

-- set torch environment
opt = {
   hdf5_files = {
     
     --'/home/yltian/3D/Data/train_HDF5_x12/02691156.h5',  --airplane
     --'/home/yltian/3D/Data/train_HDF5_x12/02828884.h5', --bench
     --'/home/yltian/3D/Data/train_HDF5_x12/02933112.h5', --cabinet
     --'/home/yltian/3D/Data/train_HDF5_x12/02958343.h5', --car
     --'/home/yltian/3D/Data/train_HDF5_x12/03001627.h5', --chair
     --'/home/yltian/3D/Data/train_HDF5_x12/04256520.h5', --sofa
     '/home/yltian/3D/Data/train_HDF5_x12/04379243.h5', --table
     
     --[[
     '/home/yltian/3D/Data/train_HDF5/02691156.h5', --airplane    2831
     '/home/yltian/3D/Data/train_HDF5/02828884.h5', --bench       1269
     '/home/yltian/3D/Data/train_HDF5/02933112.h5', --cabinet     1099
     '/home/yltian/3D/Data/train_HDF5/02958343.h5', --car         2473
     '/home/yltian/3D/Data/train_HDF5/03001627.h5', --chair       4744
     '/home/yltian/3D/Data/train_HDF5/04256520.h5', --sofa        2221
     '/home/yltian/3D/Data/train_HDF5/04379243.h5', --table       5905
     --]]
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
   folder = 'checkpoints_MVGAN'
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
local errD, errG
local epoch_tm = torch.Timer()
local tm = torch.Timer()
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
   end
end

local VolumetricConv = nn.VolumetricConvolution
local VolumetricFullConv = nn.VolumetricFullConvolution
local VolumetricBN = nn.VolumetricBatchNormalization
local VolumetricProj = nn.VolumetricProjection

local SpatialConv = nn.SpatialConvolution
local SpatialBN = nn.SpatialBatchNormalization

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

netG:apply(weights_init)

local netD = nn.Sequential()

netD:add(VolumetricProj(0.5))
netD:add(SpatialConv(nc * 6, ndf, 4, 4, 2, 2, 1, 1))
netD:add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConv(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBN(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConv(ndf * 2, ndf * 4, 4, 4, 2, 2, 1, 1))
netD:add(SpatialBN(ndf * 4)):add(nn.LeakyReLU(0.2, true))
netD:add(SpatialConv(ndf * 4, 1, 4, 4))
netD:add(nn.Sigmoid())
netD:add(nn.View(1):setNumInputDims(3))

netD:apply(weights_init)

local criterion = nn.BCECriterion()

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
-----------------------------------------------------
if opt.gpu > 0 then
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   input = input:cuda();  noise = noise:cuda();  label = label:cuda()

   if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(netG, cudnn)
      cudnn.convert(netD, cudnn)
   end
   netD:cuda();           netG:cuda();           criterion:cuda()
end

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

local fDx = function(x)
  gradParametersD:zero()
   
  -- train with real
  data_tm:reset(); data_tm:resume()
  local real = sampler:sampleBalance(opt.batch_size)
  real = torch.reshape(real, torch.LongStorage{opt.batch_size, 1, 30, 30, 30})
  data_tm:stop()
  input:fill(0)
  input[{{},{},{2,31},{2,31},{2,31}}]:copy(real)
  label:fill(real_label)

  local output = netD:forward(input)
  local errD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(input, df_do)
  
  -- train with fake
  noise:normal(0, 1)
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(fake_label)

  local output = netD:forward(input)
  local errD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netD:backward(input, df_do)

  errD = errD_real + errD_fake

  return errD, gradParametersD
end

local fGx = function(x)
  gradParametersG:zero()
  
  noise:normal(0, 1)
  local fake = netG:forward(noise)
  input:copy(fake)
  label:fill(real_label)
  
  local output = netD:forward(input)
  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(input, df_do)
  
  netG:backward(noise, df_dg)
  return errG, gradParametersG
end

local fGx2 = function(x)
  gradParametersG:zero()
  --[[ the three lines below were already executed in fDx, so save computation
  noise:normal(0, 1) -- regenerate random noise
  local fake = netG:forward(noise)
  input:copy(fake) ]]--
  label:fill(real_label)
  
  local output = netD.output
  errG = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  local df_dg = netD:updateGradInput(input, df_do)
  
  netG:backward(noise, df_dg)
  return errG, gradParametersG
end


for epoch = 1, opt.nepoch do
  epoch_tm:reset()
  local counter = 0
  for i = 1, math.min(sampler:size(), opt.ntrain), opt.batch_size do
  --for i = 1, 40, opt.batch_size do
    tm:reset()
    
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    for k = 1, opt.D_k do
      optim.adam(fDx, parametersD, optimStateD)
    end
    -- (2) Update G network: maximize log(D(G(z)))
    optim.adam(fGx2, parametersG, optimStateG)
    for k = 2, opt.G_k do
      optim.adam(fGx, parametersG, optimStateG)
    end
    
    -- logging
      if ((i-1) / opt.batch_size) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batch_size),
                 math.floor(math.min(sampler:size(), opt.ntrain) / opt.batch_size),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errD and errD or -1))
      end
  end
  
  paths.mkdir(opt.folder)
  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil
  torch.save(opt.folder..'/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
  torch.save(opt.folder..'/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
  parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = netG:getParameters()
  print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.nepoch, epoch_tm:time().real))
  
end