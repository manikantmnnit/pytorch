import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameters
input_size=28
seqence_length=28
num_layers=2
hidden_size=256
num_classes=10
learning_rate=.001
batch_size=64
num_epochs=4
load_model=True

# create RNN
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super(RNN,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size*seqence_length,num_classes)

    def forward(self,x):
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)

        # forward
        out,_=self.rnn(x,h0)
        out= out.reshape(out.shape[0],-1)
        out=self.fc(out)
        return out
def save_checkpoint(state,filename="my_checkpoint_pth.tar"):
    print("------> Saving checkpoint")
    torch.save(state,filename)

def load_chechpoint(checkpoint):
    print("-----> loading checkpoint")
    model1.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


#load data
train_data=datasets.MNIST(root='datasets/',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_data=datasets.MNIST(root='datasets/',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True)

# initialize network
model1=RNN(input_size,hidden_size,num_layers,num_classes).to(device)

# loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model1.parameters(),lr=learning_rate)

if load_model:
    load_chechpoint(torch.load('my_checkpoint_pth.tar'))
# train network
for epoch in range(num_epochs):
    loss=[]

    if epoch==2:
        checkpoint=({'state_dict':model1.state_dict(),'optimizer':optimizer.state_dict()})
        save_checkpoint(checkpoint)


    for batch_idx, (data, targets) in enumerate(train_loader):
        # Reshape the data from (batch_size, 1, 28, 28) to (batch_size, 28, 28)
        data = data.to(device=device).squeeze(1)  # (batch_size, 1, 28, 28) -> (batch_size, 28, 28)
        targets = targets.to(device=device)

        # Forward pass
        scores = model1(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent
        optimizer.step()

# check accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            # Reshape the data from (batch_size, 1, 28, 28) to (batch_size, 28, 28)
            x = x.to(device=device).squeeze(1)  # (batch_size, 1, 28, 28) -> (batch_size, 28, 28)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model1)
check_accuracy(test_loader, model1)




