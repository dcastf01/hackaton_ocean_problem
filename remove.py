import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader
target=torch.tensor(([3], [3],[3]),dtype=torch.long)

input = torch.tensor([[1.,0.,0.,0.], [0.,0.,1.,0.],[0.,1.,0.,0.]],requires_grad=True)
target=target.squeeze()
# target = torch.empty(3, dtype=torch.long).random_(5)
nClasses=4
weights = torch.ones(nClasses)
ignore_classes = torch.LongTensor([3])
weights[ignore_classes] = 0.0
loss = nn.CrossEntropyLoss(weights)
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)

output = loss(input, target)
print(input)
print(target)
print(weights)
print(output)
# output.backward()