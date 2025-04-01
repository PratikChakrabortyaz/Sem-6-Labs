import torch
import torch.nn as nn
import torch.nn.functional as F

data = torch.arange(1, 21, dtype=torch.float32)

seq_length = 5
batch_size = 4
input_size = 1

inputs = data.view(batch_size, seq_length, input_size)

hidden_size = 1
num_layers = 1
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

for name, param in lstm.named_parameters():
    nn.init.constant_(param, 1.0)

h0 = torch.zeros(num_layers, batch_size, hidden_size)
c0 = torch.zeros(num_layers, batch_size, hidden_size)

output, (hn, cn) = lstm(inputs, (h0, c0))

print("Input:")
print(inputs.squeeze(-1))
print("\nLSTM Output (last hidden state for each timestep):")
print(output.squeeze(-1))
print("\nFinal Hidden State hn:")
print(hn.squeeze())
print("\nFinal Cell State cn:")
print(cn.squeeze())
"""Input:
tensor([[ 1.,  2.,  3.,  4.,  5.],
        [ 6.,  7.,  8.,  9., 10.],
        [11., 12., 13., 14., 15.],
        [16., 17., 18., 19., 20.]])

LSTM Output (last hidden state for each timestep):
tensor([[0.7038, 0.9501, 0.9916, 0.9983, 0.9996],
        [0.7612, 0.9639, 0.9950, 0.9993, 0.9999],
        [0.7616, 0.9640, 0.9951, 0.9993, 0.9999],
        [0.7616, 0.9640, 0.9951, 0.9993, 0.9999]], grad_fn=<SqueezeBackward1>)

Final Hidden State hn:
tensor([0.9996, 0.9999, 0.9999, 0.9999], grad_fn=<SqueezeBackward0>)

Final Cell State cn:
tensor([4.9173, 4.9995, 5.0000, 5.0000], grad_fn=<SqueezeBackward0>)

"""