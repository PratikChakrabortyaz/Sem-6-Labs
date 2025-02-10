import torch
t1=torch.randn(2,3)
t2=torch.randn(2,3)
new_t=torch.matmul(t1,t2.T)
print(new_t)