import torch
x=torch.tensor([2.0],requires_grad=True)
y=torch.tensor([3.0],requires_grad=True)
z=torch.tensor([4.0],requires_grad=True)
a=2*x
b=torch.sin(y)
c=a/b
d=c*z
e=torch.log(d+1)
f=torch.tanh(e)
f.backward()
print(f"a: {a}")
print(f"b: {b}")
print(f"c: {c}")
print(f"d: {d}")
print(f"e: {e}")
def diff1(a,b,d,e,y,z):
    df_de = 1 - torch.tanh(e) ** 2
    de_dd = 1 / (d + 1)
    dd_dc = z
    dc_db = -a / (b ** 2)
    db_dy = torch.cos(y)
    df_dy = df_de * de_dd * dd_dc * dc_db * db_dy
    return df_dy
print(f"Analytical Solution:{diff1(a,b,d,e,y,z)}")
print(f"PyTorch Solution:{y.grad.item()}")

