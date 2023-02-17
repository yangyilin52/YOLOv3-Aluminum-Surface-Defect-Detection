import torch

baseT = torch.tensor([10, 11, 12])
x = torch.full((3, 3), baseT)
print(x)
