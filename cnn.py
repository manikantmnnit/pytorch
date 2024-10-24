import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

import torchvision.transforms as transforms
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # First fully connected layer
        self.fc2 = nn.Linear(50, num_classes)  # Second fully connected layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after fc1
        x = self.fc2(x)
        return x

class CNN_model(nn.Module):
    def __init__(self,in_channel=1,num_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1=nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        return x


# x=torch.randn(64,1,28,28)
# model2=CNN_model()
# print(model2(x).shape)
# set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channel=1
num_classes=10
learning_rate=.001
batch_size=64
num_echos=1

#load data
train_data=datasets.MNIST(root='datasets/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_data=datasets.MNIST(root='datasets/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

# initialize network
model1=CNN_model().to(device)

# loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model1.parameters(),lr=learning_rate)

# train network
for epoch in range(num_echos):
    for batch_idx,(data,targets) in enumerate(train_loader):
        data=data.to(device=device)
        targets=targets.to(device=device)

        # data=data.reshape(data.shape[0],-1)

        # forward
        scores=model1(data)
        loss=criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient decent
        optimizer.step()

# check accuracy
def check_accuracy(loader,model):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)
            # x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'Got{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:02f}')

    model1.train()

check_accuracy(train_loader,model1)
check_accuracy(test_loader,model1)


