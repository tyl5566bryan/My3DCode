local VolumetricProjection, parent = torch.class('nn.VolumetricProjection', 'nn.Module')

function VolumetricProjection:__init(t)
  parent.__init(self)
  self.t = t or 0.5
  -- for the purpose of CUDA support, all local variables used 
  -- in <updateOutput> and <updateGradInput> should be declared
  -- in the __init function
  self.mask    = torch.Tensor()
  self.map     = torch.Tensor()
  self.occupy  = torch.Tensor()
end

function VolumetricProjection:updateOutput(input)
  if input:dim() ~= 5 then
    error('Input must be 5D (nbatch, nfeat, t, h, w)')
  end
  if input:size(3)~=input:size(4) or input:size(4) ~= input:size(5) then
    error('Input must be cubes')
  end

  self.output:resize(torch.LongStorage{input:size(1), 6, input:size(3), input:size(3)}):fill(0)
  self.mask:resize(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  self.map:resize(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  local mylen = input:size(3)
  local myindex = torch.linspace(mylen, 1, mylen):long()
  
  for i = 1, 6 do
    local proj = self.output[{{},{i},{},{}}]
    for j = 1, mylen do
      self.mask:fill(0)
      if i == 1 then
        self.map:copy(input[{{},{},{j},{},{}}])
      elseif i == 2 then
        self.map:copy(input[{{},{},{mylen-j+1},{},{}}]:index(5, myindex))
      elseif i == 3 then
        self.map:copy(input[{{},{},{},{j},{}}])
      elseif i == 4 then
        self.map:copy(input[{{},{},{},{mylen-j+1},{}}]:index(5, myindex))
      elseif i == 5 then
        self.map:copy(input[{{},{},{},{},{j}}])
      else
        self.map:copy(input[{{},{},{},{},{mylen-j+1}}]:index(4, myindex))
      end
      self.mask:copy(torch.cmul(torch.lt(proj, self.t), torch.ge(self.map, self.t)):typeAs(self.mask))
      proj:addcmul(1, self.mask, self.map)
    end
  end
  
  return self.output
end

function VolumetricProjection:updateGradInput(input, gradOutput)
  if input:dim() ~= 5 or gradOutput:dim() ~= 4 then
    error('Input must be 5D and gradOutput must be 4D')
  end
  if input:size(3)~=input:size(4) or input:size(4)~=input:size(5) then
    error('Input must be cubes')
  end
  if gradOutput:size(3) ~= gradOutput:size(4) then
    error('GradOutput must be square')
  end
  if gradOutput:size(3) ~= input:size(3) then
    error('Input and GradOutput don\'t match')
  end
  
  self.gradInput:resizeAs(input):fill(0)
  self.mask:resize(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  self.occupy:resize(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  self.map:resize(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  local mylen = input:size(3)
  local myindex = torch.linspace(mylen, 1, mylen):long()
  
  for i = 1, 6 do
    self.occupy:fill(0)
    local proj = gradOutput[{{},{i},{},{}}]
    for j = 1, mylen do
      self.mask:fill(0)
      local gradMap
      if i == 1 then
        self.map:copy(input[{{},{},{j},{},{}}])
        gradMap = self.gradInput[{{},{},{j},{},{}}]
      elseif i == 2 then
        self.map:copy(input[{{},{},{mylen-j+1},{},{}}]:index(5, myindex))
        gradMap = self.gradInput[{{},{},{mylen-j+1},{},{}}]
      elseif i == 3 then
        self.map:copy(input[{{},{},{},{j},{}}])
        gradMap = self.gradInput[{{},{},{},{j},{}}]
      elseif i == 4 then
        self.map:copy(input[{{},{},{},{mylen-j+1},{}}]:index(5, myindex))
        gradMap = self.gradInput[{{},{},{},{mylen-j+1},{}}]
      elseif i == 5 then
        self.map:copy(input[{{},{},{},{},{j}}])
        gradMap = self.gradInput[{{},{},{},{},{j}}]
      else
        self.map:copy(input[{{},{},{},{},{mylen-j+1}}]:index(4, myindex))
        gradMap = self.gradInput[{{},{},{},{},{mylen-j+1}}]
      end
      self.mask = torch.cmul(torch.lt(self.occupy, 0.5), torch.ge(self.map, self.t)):typeAs(self.mask)
      if i % 2 == 1 then
        gradMap:addcmul(1, self.mask, proj)
      else
        gradMap:addcmul(1, self.mask:index(4, myindex), proj:index(4, myindex))
      end
      self.occupy:add(self.mask)
    end
  end
  return self.gradInput
end

function VolumetricProjection:clearState()
  return parent.clearState(self)
end