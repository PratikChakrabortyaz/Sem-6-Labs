import torch
t1=torch.randn(2,3)
t2=torch.randn(2,3)
print(t1.cuda())
print(t2.cuda())