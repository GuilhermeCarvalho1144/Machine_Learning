import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter


# import pdb

writer = SummaryWriter('runs/mnist')

#define device
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#hyperparameters
input_size = 28*28
hidden_size = 100
num_classes = 10
n_epochs = 20
batch_size = 4096
lr = 0.001
PATH = 'best_models.pth'


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.layer1(x)
        output = self.relu(output)
        output = self.layer2(output)
        #do not apply softmax, the loss will apply it
        return output


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) #input:28x28 output:24x24 
        self.pool = nn.MaxPool2d(2,2) #input:24x24 output:12x12
        self.conv2 = nn.Conv2d(6,16,5)#input:12x12 output:9X9
        self.fc1 = nn.Linear(16*4*4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #input:28x28; after conv:24x24 after pool: 12x12
        x = self.pool(F.relu(self.conv2(x))) #input:12x12; after conv:9x9 after pool: 4x4
        x = x.view(-1,16*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# dataset
def get_dataloaders():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset= torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    # examples = iter(train_loader)
    # samples, labels = next(examples) 
    return train_loader, test_loader


def train_fn(model, train_loader, criterion, opt, ):
    mean_loss = 0.0
    epoch_acc = 0

    #traning loop
    for epoch in range(n_epochs):
        for i, (img, labels) in enumerate(train_loader):
            # breakpoint()       
            #print('IMAGE shape:', img.shape)
            #print('LABELS shape', labels.shape)

            #reshape img
            img = img.to(device)
            labels = labels.to(device)

            #print('IMAGE shape:', img.shape)
            #foward step
            y_pred = model(img)
            loss = criterion(y_pred, labels)
            
            #backpass
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            mean_loss += loss.item()
            _, pred = torch.max(y_pred, 1)
            epoch_acc += (pred == labels).sum().item()
            #Prints
            if (i+1) % int(len(train_loader)/10)==0:
                print(f'epoch: {epoch+1}/{n_epochs}, step {i+1}/{len(train_loader)}, loss: {loss.item():.4f}')
                writer.add_scalar('training loss', mean_loss/100, epoch*len(train_loader)+i)
                writer.add_scalar('accuracy', epoch_acc/100, epoch*len(train_loader)+i)
                mean_loss = 0.0
                epoch_acc = 0
    torch.save(model, PATH)

def train():
    #model = MLP(input_size, hidden_size, num_classes)
    model = CNN(num_classes)
    model.to(device)

    #loss and optmizer
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader, _ = get_dataloaders()
    train_fn(model, train_loader, criterion, opt)


#test
def test():
    model = torch.load(PATH).to(device)
    _, test_loader = get_dataloaders()
    with torch.no_grad():
        n_correct = 0 
        n_samples = 0
        for img, y_true in test_loader:
            img = img.to(device)
            y_true = y_true.to(device)
            model.to(device)
            y_pred = model(img)

            #value, index
            _, pred = torch.max(y_pred,1)
            n_samples += y_true.shape[0]
            n_correct += (pred == y_true).sum().item()
            
        acc = 100.0*n_correct/n_samples

    print(f'accuracy: {acc}%')

if __name__ == "__main__":
    train()
    test()
