require 'torch'
require 'nn'
require 'hdf5'

opt = {
    batch_size = 64,        -- number of samples to produce
    net = './checkpoints/3DShape_24_net_G.t7',   -- path to the generator network
    noise_mode = 'random',  -- random / line / linefull1d / linefull
    name = '3Dgeneration.h5',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,              
}

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
end

local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')

local noise = torch.Tensor(opt.batch_size, opt.nz, 1, 1, 1)
local sample_input = torch.Tensor(2, opt.nz, 1, 1, 1)
local net = torch.load(opt.net)

noise:normal(0, 1)
sample_input:normal(0, 1)
-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    cudnn.convert(net, cudnn)
    noise = noise:cuda()
    sample_input = sample_input:cuda()
end

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(net, sample_input)

local output = net:forward(noise)

local res = torch.Tensor(output:size())
res:copy(output)

local myFile = hdf5.open(opt.name, 'w')
myFile:write('data', res)
myFile:close()





