import torch
t1=torch.randn(2,3)
t2=torch.randn(2,3)
new_t=torch.matmul(t1,t2.T)
print(f"Max value is:{torch.max(new_t)}")
print(f"Min value is:{torch.min(new_t)}")
print(f"Max index value is:{torch.argmax(new_t)}")
print(f"Min index value is:{torch.argmin(new_t)}")