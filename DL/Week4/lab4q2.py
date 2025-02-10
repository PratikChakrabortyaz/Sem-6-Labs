import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np


loss_list = []
torch.manual_seed(42)
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([0,1,1,0], dtype=torch.float32)
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel,self).__init__()
        self.linear1=nn.Linear(2,2,bias=True)
        self.activation1=nn.ReLU()
        self.linear2=nn.Linear(2,1,bias=True)
    def forward(self,x):
        x=self.linear1(x)
        x=self.activation1(x)
        x=self.linear2(x)
        return x
class MyDataset(Dataset):
    def __init__(self,X,Y):
        self.X=X
        self.Y=Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        return self.X[idx].to(device),self.Y[idx].to(device)
full_dataset=MyDataset(X,Y)
batch_size=1
train_data_loader=DataLoader(full_dataset,batch_size=batch_size,shuffle=True)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=XORModel().to(device)
print(model)
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.03)
def train_one_epoch(epoch_index):
    totalloss=0
    for i,data in enumerate(train_data_loader):
        inputs,labels=data
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=loss_fn(outputs.flatten(),labels)
        loss.backward()
        optimizer.step()
        totalloss+=loss.item()
    return totalloss/(len(train_data_loader)*batch_size)

epochs=10000
for epoch in range(epochs):
    model.train(True)
    avg_loss=train_one_epoch(epoch)
    loss_list.append(avg_loss)
    if epoch%1000==0:
        print(f"Epoch {epoch}/{epochs},Loss:{avg_loss}")
for param in model.named_parameters():
    print(param)
input=torch.tensor([0,1],dtype=torch.float32).to(device)
model.eval()
print("The input is = {}".format(input))
print("Output y predicted = {}".format(model(input)))
plt.plot(loss_list)
plt.show()
