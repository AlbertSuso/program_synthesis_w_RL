import torch

#Vaaaaa

a = torch.randint(0, 9, (3,))
b = torch.randint(0, 9, (3, 1))

c = a*b
print(c)