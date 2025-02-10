import torch
t1=torch.arange(0,10)
print(t1)
print(f"Reshaped tensor: {t1.reshape(2,5)}")
print(f"View of tensor:{t1.view(2,5)}")
t2=torch.tensor([1,2,3])
t3=torch.tensor([4,5,6])
stacked_t=torch.stack((t2,t3))
print(f"Stacked tensor:{stacked_t}")
t4=torch.zeros(2,1,2)
print(f"t4:{t4}")
print(f"Squeeze tensor:{torch.squeeze(t4)}")
print(f"Unsqueeze tensor:{torch.unsqueeze(t4,0)}")

