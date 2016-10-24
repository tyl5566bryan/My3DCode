require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'batchSampler'

-- set torch environment
opt = {
   hdf5_files = {
     
     --'/home/yltian/3D/Data/train_HDF5_x12/02691156.h5',  --airplane
     --'/home/yltian/3D/Data/train_HDF5_x12/02828884.h5', --bench
     --'/home/yltian/3D/Data/train_HDF5_x12/02933112.h5', --cabinet
     --'/home/yltian/3D/Data/train_HDF5_x12/02958343.h5', --car
     '/home/yltian/3D/Data/train_HDF5_x12/03001627.h5', --chair
     --'/home/yltian/3D/Data/train_HDF5_x12/04256520.h5', --sofa
     --'/home/yltian/3D/Data/train_HDF5_x12/04379243.h5', --table
     
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
   gpu = 2,
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
   folder = 'checkpoints_BiGAN'
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
local errD, errG, errE
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

local Linear = nn.Linear

local ngf = opt.ngf
local ndf = opt.ndf
local nz = opt.nz
local nc = opt.nc

-- generator p(x|z)
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

-- encoder p(z|x)
local netE = nn.Sequential()
netE:add(VolumetricConv(nc, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netE:add(nn.LeakyReLU(0.2, true))
netE:add(VolumetricConv(ndf, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netE:add(VolumetricBN(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netE:add(VolumetricConv(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netE:add(VolumetricBN(ndf * 4)):add(nn.LeakyReLU(0.2, true))
netE:add(VolumetricConv(ndf * 4, nz, 4, 4, 4))
netE:apply(weights_init)

-- discriminator
local netDx = nn.Sequential()
netDx:add(VolumetricConv(nc, ndf, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netDx:add(nn.LeakyReLU(0.2, true))
netDx:add(VolumetricConv(ndf, ndf * 2, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netDx:add(VolumetricBN(ndf * 2)):add(nn.LeakyReLU(0.2, true))
netDx:add(VolumetricConv(ndf * 2, ndf * 4, 4, 4, 4, 2, 2, 2, 1, 1, 1))
netDx:add(VolumetricBN(ndf * 4)):add(nn.LeakyReLU(0.2, true))
netDx:add(VolumetricConv(ndf * 4, 256, 4, 4, 4))
netDx:add(nn.View(256))
netDx:apply(weights_init)

local netDz = nn.Sequential()
netDz:add(nn.View(nz))
netDz:add(Linear(nz, 256))
netDz:add(nn.LeakyReLU(0.2, true)):add(nn.Dropout(0.2))
netDz:add(Linear(256, 256))
netDz:add(nn.LeakyReLU(0.2, true)):add(nn.Dropout(0.2))
netDz:apply(weights_init)

local netDxz = nn.Sequential()
netDxz:add(nn.JoinTable(2))
netDxz:add(Linear(512, 512))
netDxz:add(nn.LeakyReLU(0.2, true)):add(nn.Dropout(0.2))
netDxz:add(Linear(512, 512))
netDxz:add(nn.LeakyReLU(0.2, true)):add(nn.Dropout(0.2))
netDxz:add(Linear(512, 1))
netDxz:add(nn.Sigmoid())
netDxz:apply(weights_init)

local inData1 = nn.Identity()()
local inData2 = nn.Identity()()
local midData1 = netDx(inData1)
local midData2 = netDz(inData2)
local outData = netDxz({midData1, midData2})
local netD = nn.gModule({inData1, inData2},{outData})

local criterion = nn.BCECriterion()

optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateE = {
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
      cudnn.convert(netE, cudnn)
      cudnn.convert(netDx, cudnn)
      cudnn.convert(netDz, cudnn)
      cudnn.convert(netDxz, cudnn)
   end
   netD:cuda(); netE:cuda(); netG:cuda(); criterion:cuda()
   
   require 'cutorch'
   cutorch.manualSeed(opt.manualSeed)
   cutorch.setDevice(opt.gpu)
   
end

local parametersD, gradParametersD = netD:getParameters()
local parametersE, gradParametersE = netE:getParameters()
local parametersG, gradParametersG = netG:getParameters()

local fDx = function(x)
  gradParametersD:zero()
  gradParametersE:zero()
  gradParametersG:zero()
  
  local output, df_do, gradX, gradZ
   
  -- train with real
  data_tm:reset(); data_tm:resume()
  local real = sampler:sampleBalance(opt.batch_size)
  real = torch.reshape(real, torch.LongStorage{opt.batch_size, 1, 30, 30, 30})
  data_tm:stop()
  input:fill(0)
  input[{{},{},{2,31},{2,31},{2,31}}]:copy(real)
  label:fill(real_label)
  local XtoZ = netE:forward(input)
  noise:copy(XtoZ)
  output = netD:forward({input, noise})
  local errD_real = criterion:forward(output, label)
  df_do = criterion:backward(output, label)
  netD:backward({input, noise}, df_do)
  
  label:fill(fake_label)
  errE = criterion:forward(output, label)
  df_do = criterion:backward(output, label)
  gradX, gradZ = unpack(netD:updateGradInput({input, noise}, df_do))
  netE:backward(input, gradZ)
  
  -- train with fake
  noise:normal(0, 1)
  local ZtoX = netG:forward(noise)
  input:copy(ZtoX)
  label:fill(fake_label)
  output = netD:forward({input, noise})
  local errD_fake = criterion:forward(output, label)
  df_do = criterion:backward(output, label)
  netD:backward({input, noise}, df_do)
  
  label:fill(real_label)
  errG = criterion:forward(output, label)
  df_do = criterion:backward(output, label)
  gradX, gradZ = unpack(netD:updateGradInput({input, noise}, df_do))
  netG:backward(noise, gradX)

  errD = errD_real + errD_fake

  return errD, gradParametersD
end

local fGx = function(x)
  return errG, gradParametersG
end

local fEx = function(x)
  return errE, gradParametersE
end


for epoch = 1, opt.nepoch do
  epoch_tm:reset()
  local counter = 0
  for i = 1, math.min(sampler:size(), opt.ntrain), opt.batch_size do
  --for i = 1, 40, opt.batch_size do
    tm:reset()
    
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    optim.adam(fDx, parametersD, optimStateD)
    -- (2) Update G network: maximize log(D(G(z)))
    optim.adam(fGx, parametersG, optimStateG)
    -- (3) Update E network: maxmize log(1-D(E(x))
    optim.adam(fEx, parametersE, optimStateE)
    
    -- logging
      if ((i-1) / opt.batch_size) % 1 == 0 then
         print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
                   .. '  Err_G: %.4f  Err_E: %.4f  Err_D: %.4f'):format(
                 epoch, ((i-1) / opt.batch_size),
                 math.floor(math.min(sampler:size(), opt.ntrain) / opt.batch_size),
                 tm:time().real, data_tm:time().real,
                 errG and errG or -1, errE and errE or -1, errD and errD or -1))
      end
  end
  
  paths.mkdir(opt.folder)
  parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
  parametersG, gradParametersG = nil, nil
  parametersE, gradParametersE = nil, nil
  torch.save(opt.folder..'/' .. opt.name .. '_' .. epoch .. '_net_G.t7', netG:clearState())
  torch.save(opt.folder..'/' .. opt.name .. '_' .. epoch .. '_net_D.t7', netD:clearState())
  torch.save(opt.folder..'/' .. opt.name .. '_' .. epoch .. '_net_E.t7', netE:clearState())
  parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
  parametersG, gradParametersG = netG:getParameters()
  parametersE, gradParametersE = netE:getParameters()
  print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.nepoch, epoch_tm:time().real))
  
end