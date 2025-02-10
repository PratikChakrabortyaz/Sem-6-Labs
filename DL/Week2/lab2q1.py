import torch
a=torch.tensor([2.0],requires_grad=True)
b=torch.tensor([3.0],requires_grad=True)
x=2*a+3*b
y=5*a**2+3*b**3
z=2*x+3*y
def diff1(a):
    dz_dx = 2
    dz_dy = 3
    dx_da = 2
    dy_da = 10 * a
    dz_da=dz_dx*dx_da+dz_dy*dy_da
    return dz_da
z.backward()
print(f"Analytical Solution:{diff1(a)}")
print(f"PyTorch Solution:{a.grad.item()}")