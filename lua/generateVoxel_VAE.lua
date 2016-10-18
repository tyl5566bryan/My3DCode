require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'

require 'Sampler'

-- setting
--------------------------------------------
opt = {
    batch_size = 64,        -- number of samples to produce
    net = './checkpoints_table_VAE/3DShape_25_VAE.t7',   -- path to the generator network
    name = '3Dgeneration_VAE.h5',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    data_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5',
    encoder = './checkpoints/3DShape_25_Encoder.t7',
    decoder = './checkpoints/3DShape_25_Decoder.t7',
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
local noise  = torch.Tensor(opt.batch_size, opt.nz)
noise:normal(0, 1)

local model = torch.load(opt.net)
local encoder = torch.load(opt.encoder)
local decoder = torch.load(opt.decoder)

if opt.gpu > 0 then
    model:cuda()
    encoder:cuda()
    decoder:cuda()
    
    cudnn.convert(model, cudnn)
    cudnn.convert(encoder, cudnn)
    cudnn.convert(decoder, cudnn)
    
    inputs = inputs:cuda()
    noise = noise:cuda()
end

-- sample output and save
--------------------------------------------
local reconstruction, mean, log_var
reconstruction, mean, log_var = unpack(model:forward(inputs))

local res = torch.Tensor(reconstruction:size())
res:copy(reconstruction)

local e_mean, e_log_var, d_reconstruction
e_mean, e_log_var = unpack(encoder:forward(inputs))
d_reconstruction = decoder:forward(e_mean)

local res2 = torch.Tensor(d_reconstruction:size())
res2:copy(d_reconstruction)

local generation = decoder:forward(noise)
local res3 = torch.Tensor(generation:size())
res3:copy(generation)

local mydata = torch.cat(res, res2, 1)
mydata = torch.cat(mydata, res3, 1)

local myFile = hdf5.open(opt.name, 'w')
myFile:write('data', mydata)
myFile:close()

