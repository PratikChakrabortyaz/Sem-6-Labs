import torch
import torch.nn as nn
import matplotlib.pyplot as plt


x = torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
                  19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2]).reshape(-1,1)
y = torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
                  16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6]).reshape(-1,1)


model = nn.Linear(in_features=1, out_features=1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


loss_list = []


for epoch in range(100):

    y_pred = model(x)


    loss = criterion(y_pred, y)
    loss_list.append(loss.item())


    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


    print(f"Epoch {epoch + 1}, w={model.weight.item()}, b={model.bias.item()}, loss={loss.item()}")


plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

