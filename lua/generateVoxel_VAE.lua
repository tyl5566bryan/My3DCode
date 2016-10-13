require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'

require 'Sampler'

-- setting
--------------------------------------------
opt = {
    batch_size = 64,        -- number of samples to produce
    net = './checkpoints/3DShape_25_VAE.t7',   -- path to the generator network
    name = '3Dgeneration_VAE.h5',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    data_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5',
}

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

torch.setdefaulttensortype('torch.FloatTensor')

-- loading data and model
--------------------------------------------

local h5_file = hdf5.open(opt.data_file, 'r')
local data = h5_file:read('/data'):all()

local inputs = torch.Tensor(data:size(1), 1, 32, 32, 32):fill(0)
inputs[{{},{},{2,31},{2,31},{2,31}}]:copy(data)

local model = torch.load(opt.net)

if opt.gpu > 0 then
    model:cuda()
    cudnn.convert(model, cudnn)
    inputs = inputs:cuda()
end

-- sample output and save
--------------------------------------------
local reconstruction, mean, log_var
reconstruction, mean, log_var = unpack(model:forward(inputs))

local res = torch.Tensor(reconstruction:size())
res:copy(reconstruction)

local myFile = hdf5.open(opt.name, 'w')
myFile:write('data', res)
myFile:close()

