import torch
x=torch.tensor([2.0],requires_grad=True)
w=torch.tensor([3.0],requires_grad=True)
b=torch.tensor([4.0],requires_grad=True)
u=w*x
v=u+b
a=torch.sigmoid(v)
def diff1(v,x):
    da_dv = (torch.sigmoid(v)) * (1 - torch.sigmoid(v))
    dv_du = 1
    du_dw = x
    da_dw = da_dv * dv_du * du_dw
    return da_dw

a.backward()
print(f"Analytical Solution:{diff1(v,x)}")
print(f"PyTorch Solution:{w.grad.item()}")