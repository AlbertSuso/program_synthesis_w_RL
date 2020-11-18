import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.tensor([[1, 2], [3, 4], [5, 6]])
shape = list(input.shape)
print(shape)
shape[0] = shape[0] - 1
print(shape)

mask = torch.randint(0, 2, shape)
mask2 = torch.ones(shape)-mask
mask = torch.cat((mask, mask2), dim=0)
print(mask)





