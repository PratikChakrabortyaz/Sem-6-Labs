import torch
t1=torch.randn(7,7)
print(f"Random tensor:{t1}")
t2=torch.randn(1,7)
new_t=torch.matmul(t1,t2.T)
print(f"New Tensor is:{new_t}")