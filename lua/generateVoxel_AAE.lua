require 'torch'
require 'nn'
require 'hdf5'


-- setting
--------------------------------------------
opt = {
    batch_size = 64,        -- number of samples to produce
    net = './checkpoints/3DShape_25_AAE.t7',   -- path to the generator network
    name = '3Dgeneration_table_AAE.h5',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,
    data_file = '/home/yltian/3D/Data/val_HDF5/04379243.h5',
    encoder = './checkpoints/3DShape_25_encoder.t7',
    decoder = './checkpoints/3DShape_25_decoder.t7',
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
local reconstruction = model:forward(inputs)

local feat = encoder:forward(inputs)
print(('feat mean: %.4f, feat var: %.4f'):format(feat:mean(), feat:var()))

local res1 = torch.Tensor(reconstruction:size())
res1:copy(reconstruction)

local generation = decoder:forward(noise)
local res2 = torch.Tensor(generation:size())
res2:copy(generation)

local res = torch.cat(res1, res2, 1)

local myFile = hdf5.open(opt.name, 'w')
myFile:write('data', res)
myFile:close()

