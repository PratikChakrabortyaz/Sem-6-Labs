import torch
import numpy as np
n1=np.array([1,2,3])
print(n1)
t1=torch.from_numpy(n1)
print(t1)
n2=t1.numpy()
print(n2)