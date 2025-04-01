import os
import string
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F

data_dir = '/home/student/Downloads/data/names'

language_names = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".txt"):
        language = filename.split('.')[0]
        with open(os.path.join(data_dir, filename), 'r') as f:
            language_names[language] = [line.strip() for line in f.readlines()]
all_unique_characters = set(string.ascii_letters + " ")

for language, names in language_names.items():
    for name in names:
        all_unique_characters.update(name)

all_characters = sorted(all_unique_characters)

char_to_index = {char: idx for idx, char in enumerate(all_characters)}
n_characters = len(all_characters)

padding_idx = len(all_characters)

label_encoder = LabelEncoder()
languages = list(language_names.keys())
label_encoder.fit(languages)

all_names = []
all_labels = []

for language, names in language_names.items():
    for name in names:
        all_names.append(name)
        all_labels.append(language)

all_labels = label_encoder.transform(all_labels)


class NameLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NameLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, cell) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output



input_size = n_characters + 1
hidden_size = 128
output_size = len(language_names)

model = NameLanguageLSTM(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


class NameDataset(Dataset):
    def __init__(self, names, labels, max_length):
        self.names = names
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        label = self.labels[idx]

        name_tensor = torch.tensor([char_to_index[char] for char in name], dtype=torch.long)

        if len(name_tensor) < self.max_length:
            padding = torch.full((self.max_length - len(name_tensor),), padding_idx, dtype=torch.long)
            name_tensor = torch.cat([name_tensor, padding])

        return name_tensor, label


max_length = max(len(name) for name in all_names)

dataset = NameDataset(all_names, all_labels, max_length)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for names, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(names)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_preds / total_preds

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

model.eval()
with torch.no_grad():
    name = "Yang"
    name_tensor = torch.tensor([char_to_index[char] for char in name], dtype=torch.long).unsqueeze(
        0)

    if len(name_tensor[0]) < max_length:
        padding = torch.full((max_length - len(name_tensor[0]),), padding_idx, dtype=torch.long)
        name_tensor = torch.cat([name_tensor, padding.unsqueeze(0)], dim=1)

    output = model(name_tensor)
    _, predicted_language_idx = torch.max(output, 1)
    predicted_language = label_encoder.inverse_transform([predicted_language_idx.item()])[0]

    print(f"The name '{name}' is most likely from the language: {predicted_language}")

"""Epoch [1/10], Loss: 1.6538, Accuracy: 0.5177
Epoch [2/10], Loss: 1.2027, Accuracy: 0.6633
Epoch [3/10], Loss: 0.8782, Accuracy: 0.7546
Epoch [4/10], Loss: 0.6956, Accuracy: 0.7966
Epoch [5/10], Loss: 0.6040, Accuracy: 0.8189
Epoch [6/10], Loss: 0.5355, Accuracy: 0.8354
Epoch [7/10], Loss: 0.4885, Accuracy: 0.8497
Epoch [8/10], Loss: 0.4435, Accuracy: 0.8617
Epoch [9/10], Loss: 0.4129, Accuracy: 0.8697
Epoch [10/10], Loss: 0.3848, Accuracy: 0.8790
The name 'Yang' is most likely from the language: Chinese

"""
