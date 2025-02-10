import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self, conv1_out_channels, conv2_out_channels, conv3_out_channels, fc1_out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, conv1_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(conv3_out_channels , fc1_out_features, bias=True),
            nn.ReLU(),
            nn.Linear(fc1_out_features, 10, bias=True)
        )

    def forward(self, x):
        features = self.net(x)
        return self.classification_head(features.view(x.size(0), -1))


def train_and_evaluate_model(conv1_out_channels, conv2_out_channels, conv3_out_channels, fc1_out_features,
                             num_epochs=10):
    model = CNNClassifier(conv1_out_channels, conv2_out_channels, conv3_out_channels, fc1_out_features).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(trainloader)
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total learnable parameters: {total_params}")

    return accuracy, total_params


configs = [
    (64, 128, 64, 20),
    (32, 64, 32, 10),
    (16, 32, 16, 5),
    (32, 64, 32, 5),
]

param_drops = []
accuracies = []

for conv1_out_channels, conv2_out_channels, conv3_out_channels, fc1_out_features in configs:
    print(f"\nTraining with conv1_out_channels={conv1_out_channels}, conv2_out_channels={conv2_out_channels}, "
          f"conv3_out_channels={conv3_out_channels}, fc1_out_features={fc1_out_features}")

    accuracy, total_params = train_and_evaluate_model(conv1_out_channels, conv2_out_channels, conv3_out_channels,
                                                      fc1_out_features, num_epochs=10)

    original_params = sum(p.numel() for p in CNNClassifier(64, 128, 64, 20).parameters() if p.requires_grad)

    param_drop_percentage = (original_params - total_params) / original_params * 100
    param_drops.append(param_drop_percentage)
    accuracies.append(accuracy)

plt.figure(figsize=(10, 6))
plt.plot(param_drops, accuracies, marker='o', linestyle='-', color='b')
plt.title('Percentage Drop in Parameters vs Accuracy')
plt.xlabel('Percentage Drop in Parameters')
plt.ylabel('Test Accuracy (%)')
plt.grid(True)
plt.show()
