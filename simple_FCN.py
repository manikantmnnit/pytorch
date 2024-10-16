import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# create fully connected network
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # First fully connected layer
        self.fc2 = nn.Linear(50, num_classes)  # Second fully connected layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation after fc1
        x = torch.sigmoid(self.fc2(x))  # Apply Sigmoid activation after fc2
        return x

# x=torch.randn(50,750)
# model1=Model(750,10)
# print(model1(x).shape)

# set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size=784
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
model1=Model(input_size=input_size,num_classes=num_classes).to(device)

# loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model1.parameters(),lr=learning_rate)

# train network
for epoch in range(num_echos):
    for batch_idx,(data,targets) in enumerate(train_loader):
        data=data.to(device=device)
        targets=targets.to(device=device)

        data=data.reshape(data.shape[0],-1)

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
            x=x.reshape(x.shape[0],-1)

            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f'Got{num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:02f}')

    model1.train()

check_accuracy(train_loader,model1)
check_accuracy(test_loader,model1)








