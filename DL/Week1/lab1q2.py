import torch
t1=torch.randn(3,4,5)
print(f"Original size:{t1.size()}")
print(f"New size:{torch.permute(t1,(2,0,1)).size()}")