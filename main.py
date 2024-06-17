import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as dataset
import torchvision.transforms as transforms

#device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#create a nural network

class NN(nn.Module):
    def __init__(self , input_size , num_classes):# as each image have 28*28=784 nodes
        super(NN , self ).__init__()
        self.fun1 = nn.Linear(input_size , 50)
        self.fun2 = nn.Linear(50 , num_classes)

    def forward(self, x):
        x = F.relu(self.fun1(x))
        x = self.fun2(x)

        return x

# train data

#hyper parameters
input_size = 784
num_class = 10
LR = 0.001
batch_size = 64
epoch = 1

#load data

training_data = dataset.MNIST(root="dataset/", train= True , transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset = training_data , batch_size=batch_size , shuffle=True)


test_data = dataset.MNIST(root="dataset/", train= False , transform=transforms.ToTensor() , download=True)
test_loader = DataLoader(dataset = test_data , batch_size=batch_size , shuffle=True)


#initialize model

model = NN(input_size , num_class)

#loss and optimization
critation = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters() , lr=LR)

#train data

for epooch in range(epoch):
    for batch_idx , (data,targets) in enumerate(train_loader):
        data = data.to(device = device )
        targets = targets.to(device=device)

        data = data.reshape(data.shape[0] , -1 )

        scores = model(data)
        loss = critation(scores , targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
#check accuracy

def checkaccurecy(loader , model):
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for x , y in loader:
            x=x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0] , -1)

            scores = model(x)
            _, pridictions = scores.max(1)

            num_correct += (pridictions==y).sum()
            num_samples += pridictions.size(0)

            print(f"got {num_correct} / {num_samples}  with accurecy {(float(num_correct)/float(num_samples))*100:.2f}")


    model.train()

checkaccurecy(train_loader , model)
checkaccurecy(test_loader , model)