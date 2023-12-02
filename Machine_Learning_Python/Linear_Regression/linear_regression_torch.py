import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#definig the model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        # define the layer
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.lin(x)


# Gen dataset
x_raw, y_raw = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=0)

x = torch.from_numpy(x_raw.astype(np.float32))
y = torch.from_numpy(y_raw.astype(np.float32))
# resize the y
y = y.view(y.shape[0], 1)

n_samples, n_features = x.shape


# Load the model
input_size = output_size = n_features
model = LinearRegression(input_size=input_size, output_size=output_size)

# define optim and loss
loss = nn.MSELoss()
lr = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# Training loop
epochs = 100

for epoch in range(epochs):
    y_pred = model.forward(x)
    l = loss(y_pred, y)
    #backprop
    l.backward()
    #update weigths  
    optimizer.step()
    #zero grad
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch {epoch+1}; loss {l.item():.4f}')

#plot model predictions
predictions = model(x).detach().numpy()
plt.plot(x_raw, y_raw, 'ro')
plt.plot(x_raw, predictions, 'b')
plt.show()
