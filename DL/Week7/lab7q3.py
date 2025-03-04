import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.utils.data import random_split
import matplotlib.pyplot as plt


class SimpleCNNWithDropout(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(SimpleCNNWithDropout, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        x = self.dropout(x)
        x = self.fc2(x)
        return x


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_data = datasets.ImageFolder(root='./cats_and_dogs_filtered/train', transform=transform)
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model_with_dropout = SimpleCNNWithDropout(dropout_prob=0.5)
model_without_dropout = SimpleCNNWithDropout(dropout_prob=0.0)

optimizer_with_dropout = optim.Adam(model_with_dropout.parameters(), lr=0.001)
optimizer_without_dropout = optim.Adam(model_without_dropout.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()


def train_model(model, optimizer, train_loader, num_epochs=10):
    model.train()
    train_loss = []
    train_acc = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss.append(running_loss / len(train_loader))
        train_acc.append(100 * correct / total)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {100 * correct / total}%")

    return train_loss, train_acc


def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


print("Training with Dropout Regularization:")
train_loss_dropout, train_acc_dropout = train_model(model_with_dropout, optimizer_with_dropout, train_loader)

print("\nTraining without Dropout Regularization:")
train_loss_no_dropout, train_acc_no_dropout = train_model(model_without_dropout, optimizer_without_dropout,
                                                          train_loader)

val_acc_with_dropout = validate_model(model_with_dropout, val_loader)
val_acc_without_dropout = validate_model(model_without_dropout, val_loader)

print(f"\nValidation Accuracy with Dropout: {val_acc_with_dropout}%")
print(f"Validation Accuracy without Dropout: {val_acc_without_dropout}%")


