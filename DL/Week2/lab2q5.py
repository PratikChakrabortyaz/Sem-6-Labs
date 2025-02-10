import torch
x=torch.tensor([2.0],requires_grad=True)
y=8*x**4+3*x**3+7*x**2+6*x+3
def diff1(x):
    dy_dx = 32 * x ** 3 + 9 * x ** 2 + 14 * x + 6
    return dy_dx


y.backward()
print(f"Analytical Solution:{diff1(x)}")
print(f"PyTorch Solution:{x.grad.item()}")