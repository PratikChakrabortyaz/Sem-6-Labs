import torch
import matplotlib.pyplot as plt
x = torch.tensor([2,4])
y = torch.tensor([20,40])
w=torch.tensor([1.0],requires_grad=True)
b=torch.tensor([1.0],requires_grad=True)
print("The parameters are {}, and {}".format(w,b))
lr=torch.tensor(0.001)
for epochs in range(2):
    y_pred=w*x+b
    loss=torch.mean((y_pred-y)**2)
    loss.backward()
    with torch.no_grad():
        w-=lr*w.grad
        b-=lr*b.grad
    w.grad.zero_()
    b.grad.zero_()
    print("The parameters are w={},b={}, and loss={}".format(w,b,loss.item()))
