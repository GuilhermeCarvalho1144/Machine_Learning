import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np


class WineDataset (Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:,[0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples



class ToTensor:
    def __call__(self, sample):
        input, target = sample
        return torch.from_numpy(input), torch.from_numpy(target)




class LogisticRegression(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output = self.linear(x)
        output = self.relu(output)
        output = self.linear2(output)
        return output

BATCH_SIZE = 32
dataset = WineDataset(transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

dataitter = iter(dataloader)
data = dataitter.next()
features, _ = data
input_size = features.shape[1]
print(type(features))
#training loop
num_epochs = 100
total_sample = len(dataset)
n_iterations = round(total_sample/BATCH_SIZE)



model = LogisticRegression(input_size=input_size, hidden_size=5, num_classes=3)

lr = 0.001
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


for epoch in range(num_epochs):
    running_loss = 0
    for i, (inputs, labels) in enumerate(dataloader):
        
        optimizer.zero_grad()
        y_pred = model.forward(inputs)
        
        l = loss(y_pred, labels)
        l.backward()
        optimizer.step()
        running_loss += l.item()

        if (i+1) % n_iterations == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations} loss {running_loss/n_iterations}')
