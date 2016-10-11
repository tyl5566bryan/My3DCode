require 'dataLoader'

local BatchSampler = torch.class('VoxelBatchSampler')

function BatchSampler:__init(files)
  self.loaders = {}
  for i = 1,#files do
    self.loaders[i] = VoxelDataLoader(files[i])
  end
  
  self.num_loaders = #self.loaders
  self.default_batch_size = 64
  
  self.nSamples = 0
  for i = 1,#self.loaders do
    self.nSamples = self.nSamples + self.loaders[i]:size()
  end
end

function BatchSampler:sample(n)
  n = n or self.default_batch_size
  shuffle = torch.randperm(self.num_loaders)
  local res = self.loaders[shuffle[1]]:sample(n)
  return res
end

function BatchSampler:sampleBalance(n)
  n = n or self.default_batch_size
  local l = #self.loaders
  local n_each = math.floor(n/l)
  local remain = n - l * n_each
  local inds = torch.randperm(l)
  local res = torch.Tensor(n, 30, 30, 30)
  local start = 1
  
  for i = 1, l do
    if i <= remain then
      local temp = self.loaders[inds[i]]:sample(n_each + 1)
      res[{{start, start + n_each},{},{},{}}]:copy(temp)
      start = start + n_each + 1
    end
    if i > remain then
      local temp = self.loaders[inds[i]]:sample(n_each)
      res[{{start, start + n_each - 1},{},{},{}}]:copy(temp)
      start = start + n_each
    end
  end
  
  return res
end

function BatchSampler:size()
  return self.nSamples
end