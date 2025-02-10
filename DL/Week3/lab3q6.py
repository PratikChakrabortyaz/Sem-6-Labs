import torch
import torch.nn as nn
import numpy as np


X = torch.tensor([[3, 8], [4, 5], [5, 7], [6, 3], [2, 1]],dtype=torch.float32)
Y = torch.tensor([-3.7,3.5,2.5,11.5,5.7], dtype=torch.float32).reshape(-1,1)



class MLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


model = MLR()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):

    Y_pred = model(X)

    loss = criterion(Y_pred, Y)

    optimizer.zero_grad()
    loss.backward()


    optimizer.step()


    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


test_input = torch.tensor([[3, 2]], dtype=torch.float32)
predicted = model(test_input)


print(f"Predicted value for X1=3, X2=2: {predicted.item()}")


print("Learned weights:", model.linear.weight)
print("Learned bias:", model.linear.bias)
