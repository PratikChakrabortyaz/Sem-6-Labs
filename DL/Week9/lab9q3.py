import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

text = (
    "Today is a beautiful morning. The sun is shining and the birds are chirping"
)

vocab = sorted(set(text))
vocab_size = len(vocab)
print("Vocabulary:", vocab)

char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

seq_length = 10
inputs = []
targets = []

for i in range(len(text) - seq_length):
    seq = text[i: i + seq_length]
    target = text[i + seq_length]
    inputs.append([char_to_idx[ch] for ch in seq])
    targets.append(char_to_idx[target])

inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

dataset = TensorDataset(inputs, targets)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class NextCharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(NextCharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        embed = self.embedding(x)
        out, hidden = self.lstm(embed, hidden)
        out = out[:, -1, :]
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)

embed_size = 32
hidden_size = 128
num_layers = 1
num_epochs = 200
learning_rate = 0.003

model = NextCharLSTM(vocab_size, embed_size, hidden_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    for batch_inputs, batch_targets in dataloader:
        batch_size = batch_inputs.size(0)
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        outputs, hidden = model(batch_inputs, hidden)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

def predict_next_char(model, seed_str, char_to_idx, idx_to_char, predict_len=1):
    model.eval()
    input_seq = torch.tensor([[char_to_idx[ch] for ch in seed_str]], dtype=torch.long)
    hidden = model.init_hidden(1)
    predicted = seed_str
    for _ in range(predict_len):
        output, hidden = model(input_seq, hidden)
        prob = torch.softmax(output, dim=1).data
        top_idx = torch.multinomial(prob, 1)[0]
        next_char = idx_to_char[top_idx.item()]
        predicted += next_char

        new_input = list(input_seq.squeeze().numpy())
        new_input.append(top_idx.item())
        new_input = new_input[1:]
        input_seq = torch.tensor([new_input], dtype=torch.long)
    return predicted

seed = "Today is a"
generated_text = predict_next_char(model, seed, char_to_idx, idx_to_char, predict_len=50)
print("\nGenerated text:")
print(generated_text)
"""Vocabulary: [' ', '.', 'T', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'y']
Epoch 20/200, Loss: 0.5746
Epoch 40/200, Loss: 0.0840
Epoch 60/200, Loss: 0.0365
Epoch 80/200, Loss: 0.0115
Epoch 100/200, Loss: 0.0082
Epoch 120/200, Loss: 0.0053
Epoch 140/200, Loss: 0.0045
Epoch 160/200, Loss: 0.0032
Epoch 180/200, Loss: 0.0023
Epoch 200/200, Loss: 0.0025

Generated text:
Today is a beautiful arn ching an ind s hiring and t birping
"""