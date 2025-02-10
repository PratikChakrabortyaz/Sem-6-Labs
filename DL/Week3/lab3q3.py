import torch
import matplotlib.pyplot as plt
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
lr = torch.tensor(0.001)
class RegressionModel:
    def __init__(self):
        self.w=torch.randn([1],requires_grad=True)
        self.b=torch.randn([1],requires_grad=True)
    def forward(self,x):
        return self.w*x+self.b
    def update(self):
        self.w-=lr*self.w.grad
        self.b-=lr*self.b.grad
    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()
def criterion(yj,y_p):
    return (yj-y_p)**2
model=RegressionModel()
loss_list=[]
for epochs in range(100):
    y_p=model.forward(x)
    loss=torch.mean(criterion(y,y_p))
    loss_list.append(loss.item())
    loss.backward()
    with torch.no_grad():
        model.update()
    model.reset_grad()
    print("The parameters are w={},b={}, and loss={}".format(model.w, model.b, loss.item()))
plt.plot(loss_list)
plt.show()

