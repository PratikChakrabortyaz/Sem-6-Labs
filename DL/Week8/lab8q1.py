import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
df = pd.read_csv("/home/student/Downloads/daily.csv")
df = df.dropna()
y = df['Price'].values
x = np.arange(1, len(y), 1)
print(len(y))
minm = y.min()
maxm = y.max()
print(minm, maxm)
y = (y - minm) / (maxm - minm)
Sequence_Length = 10
X = []
Y = []
for i in range(0, 5900):
    list1 = []
    for j in range(i, i + Sequence_Length):
        list1.append(y[j])
    X.append(list1)
    Y.append(y[j + 1])
X = np.array(X)
Y = np.array(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y,
test_size=0.10, random_state=42, shuffle=False, stratify=None)
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len
dataset = NGTimeSeries(x_train,y_train)
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset,shuffle=True,batch_size=256)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel,self).__init__()
        self.rnn = nn.RNN(input_size=1,hidden_size=5,num_layers=1,batch_first=True)
        self.fc1 = nn.Linear(in_features=5,out_features=1)
    def forward(self,x):
        output,_status = self.rnn(x)
        output = output[:,-1,:]
        output = self.fc1(torch.relu(output))
        return output
model = RNNModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1500

for i in range(epochs):
    for j, data in enumerate(train_loader):
        y_pred = model(data[:][0].view(-1, Sequence_Length,1)).reshape(-1)
        loss = criterion(y_pred, data[:][1])
        loss.backward()
        optimizer.step()
    if i % 50 == 0:
        print(i, "th iteration : ", loss)
test_set = NGTimeSeries(x_test,y_test)
test_pred = model(test_set[:][0].view(-1,10,1)).view(-1)
plt.plot(test_pred.detach().numpy(),label='predicted')
plt.plot(test_set[:][1].view(-1),label='original')
plt.legend()
plt.show()
y = y * (maxm - minm) + minm
y_pred = test_pred.detach().numpy() * (maxm - minm) + minm
plt.plot(y)
plt.plot(range(len(y)-len(y_pred), len(y)), y_pred)
plt.show()

''' th iteration :  tensor(0.0292, grad_fn=<MseLossBackward0>)
50 th iteration :  tensor(0.0094, grad_fn=<MseLossBackward0>)
100 th iteration :  tensor(0.0103, grad_fn=<MseLossBackward0>)
150 th iteration :  tensor(0.0109, grad_fn=<MseLossBackward0>)
200 th iteration :  tensor(0.0036, grad_fn=<MseLossBackward0>)
250 th iteration :  tensor(0.0008, grad_fn=<MseLossBackward0>)
300 th iteration :  tensor(0.0009, grad_fn=<MseLossBackward0>)
350 th iteration :  tensor(0.0018, grad_fn=<MseLossBackward0>)
400 th iteration :  tensor(0.0019, grad_fn=<MseLossBackward0>)
450 th iteration :  tensor(0.0026, grad_fn=<MseLossBackward0>)
500 th iteration :  tensor(0.0028, grad_fn=<MseLossBackward0>)
550 th iteration :  tensor(0.0021, grad_fn=<MseLossBackward0>)
600 th iteration :  tensor(0.0007, grad_fn=<MseLossBackward0>)
650 th iteration :  tensor(0.0004, grad_fn=<MseLossBackward0>)
700 th iteration :  tensor(0.0011, grad_fn=<MseLossBackward0>)
750 th iteration :  tensor(0.0001, grad_fn=<MseLossBackward0>)
800 th iteration :  tensor(0.0045, grad_fn=<MseLossBackward0>)
850 th iteration :  tensor(0.0061, grad_fn=<MseLossBackward0>)
900 th iteration :  tensor(0.0057, grad_fn=<MseLossBackward0>)
950 th iteration :  tensor(0.0010, grad_fn=<MseLossBackward0>)
1000 th iteration :  tensor(0.0041, grad_fn=<MseLossBackward0>)
1050 th iteration :  tensor(0.0079, grad_fn=<MseLossBackward0>)
1100 th iteration :  tensor(0.0081, grad_fn=<MseLossBackward0>)
1150 th iteration :  tensor(0.0064, grad_fn=<MseLossBackward0>)
1200 th iteration :  tensor(0.0047, grad_fn=<MseLossBackward0>)
1250 th iteration :  tensor(0.0030, grad_fn=<MseLossBackward0>)
1300 th iteration :  tensor(0.0005, grad_fn=<MseLossBackward0>)
1350 th iteration :  tensor(0.0012, grad_fn=<MseLossBackward0>)
1400 th iteration :  tensor(0.0035, grad_fn=<MseLossBackward0>)
1450 th iteration :  tensor(0.0054, grad_fn=<MseLossBackward0>)'''