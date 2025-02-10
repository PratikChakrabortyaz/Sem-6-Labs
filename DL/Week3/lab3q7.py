import torch
import torch.nn as nn

X = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float32).reshape(-1,1)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32).reshape(-1,1)



class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))



model = LogisticRegressionModel()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


num_epochs = 1000

for epoch in range(num_epochs):

    y_pred = model(X)


    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")


with torch.no_grad():
    predicted = model(X)
    predicted_classes = (predicted >= 0.5).float()

print(f"Predicted values: {predicted_classes.squeeze().numpy()}")
print(f"True labels: {y.squeeze().numpy()}")