import torch
x=torch.tensor([6.0],requires_grad=True)
f=torch.exp(-x**2-2*x-torch.sin(x))
def diff1(x):
    df_dx = torch.exp(-x ** 2 - 2 * x - torch.sin(x)) * (-2 * x - 2 - torch.cos(x))
    return df_dx

f.backward()
print(f"Analytical Solution:{diff1(x)}")
print(f"PyTorch Solution:{x.grad.item()}")