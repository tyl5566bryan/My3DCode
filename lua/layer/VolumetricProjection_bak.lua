local VolumetricProjection, parent = torch.class('nn.VolumetricProjection', 'nn.Module')

function VolumetricProjection:__init(t)
  parent.__init(self)
  self.t = t or 0.5
  
  -- for the purpose of CUDA support, all local variables used 
  -- in <updateOutput> and <updateGradInput> should be declared
  -- in the __init function
end

function VolumetricProjection:updateOutput(input)
  if input:dim() ~= 5 then
    error('Input must be 5D (nbatch, nfeat, t, h, w)')
  end
  if input:size(3)~=input:size(4) or input:size(4) ~= input:size(5) then
    error('Input must be cubes')
  end

  self.output:resize(torch.LongStorage{input:size(1), 6, input:size(3), input:size(3)}):fill(0)
  local mask = torch.Tensor(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  local map = torch.Tensor(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  local mylen = input:size(3)
  local myindex = torch.linspace(mylen, 1, mylen):long()
  
  for i = 1, 6 do
    local proj = self.output[{{},{i},{},{}}]
    for j = 1, mylen do
      mask:fill(0)
      if i == 1 then
        map:copy(input[{{},{},{j},{},{}}])
      elseif i == 2 then
        map:copy(input[{{},{},{mylen-j+1},{},{}}]:index(5, myindex))
      elseif i == 3 then
        map:copy(input[{{},{},{},{j},{}}])
      elseif i == 4 then
        map:copy(input[{{},{},{},{mylen-j+1},{}}]:index(5, myindex))
      elseif i == 5 then
        map:copy(input[{{},{},{},{},{j}}])
      else
        map:copy(input[{{},{},{},{},{mylen-j+1}}]:index(4, myindex))
      end
      mask:copy(torch.cmul(torch.lt(proj, self.t), torch.ge(map, self.t)):typeAs(mask))
      proj:addcmul(1, mask, map)
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
  local mask = torch.Tensor(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})   --indicate backpropogate map of current slice
  local occupy = torch.Tensor(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)}) --indicate whether each gradOutput unit has been backpropogate
  local map = torch.Tensor(torch.LongStorage{input:size(1), 1, input:size(3), input:size(3)})
  local mylen = input:size(3)
  local myindex = torch.linspace(mylen, 1, mylen):long()
  
  for i = 1, 6 do
    occupy:fill(0)
    local proj = gradOutput[{{},{i},{},{}}]
    for j = 1, mylen do
      mask:fill(0)
      local gradMap
      if i == 1 then
        map:copy(input[{{},{},{j},{},{}}])
        gradMap = self.gradInput[{{},{},{j},{},{}}]
      elseif i == 2 then
        map:copy(input[{{},{},{mylen-j+1},{},{}}]:index(5, myindex))
        gradMap = self.gradInput[{{},{},{mylen-j+1},{},{}}]
      elseif i == 3 then
        map:copy(input[{{},{},{},{j},{}}])
        gradMap = self.gradInput[{{},{},{},{j},{}}]
      elseif i == 4 then
        map:copy(input[{{},{},{},{mylen-j+1},{}}]:index(5, myindex))
        gradMap = self.gradInput[{{},{},{},{mylen-j+1},{}}]
      elseif i == 5 then
        map:copy(input[{{},{},{},{},{j}}])
        gradMap = self.gradInput[{{},{},{},{},{j}}]
      else
        map:copy(input[{{},{},{},{},{mylen-j+1}}]:index(4, myindex))
        gradMap = self.gradInput[{{},{},{},{},{mylen-j+1}}]
      end
      mask = torch.cmul(torch.lt(occupy, 0.5), torch.ge(map, self.t)):typeAs(mask)
      if i % 2 == 1 then
        gradMap:addcmul(1, mask, proj)
      else
        gradMap:addcmul(1, mask:index(4, myindex), proj:index(4, myindex))
      end
      occupy:add(mask)
    end
  end
  return self.gradInput
end