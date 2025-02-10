import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])



class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(device), self.y[idx].to(device)


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.w * x + self.b


model = RegressionModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

dataset = MyDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

loss_list = []
for epoch in range(100):
    total_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    loss_list.append(total_loss / len(dataloader))
    print(f"Epoch {epoch + 1}, w={model.w.item()}, b={model.b.item()}, loss={total_loss / len(dataloader)}")

plt.plot(loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.show()

