import torch
torch.manual_seed(7)
t1=torch.randn(1,1,1,10)
t2=torch.squeeze(t1)
print(f"First tensor:{t1}")
print(f"First tensor's shape:{t1.shape}")
print(f"Second tensor:{t2}")
print(f"Second tensor's shape:{t2.shape}")