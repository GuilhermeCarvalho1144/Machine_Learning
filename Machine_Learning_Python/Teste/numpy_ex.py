import torch
import numpy as np
'''
Linear regression(LR) using numpy

We are going to do a linear regression ussing only numpy

f = w*x

For this example we will keep it simple
the focus here is not learn LR
the foucs in on t learning pytorch
'''

# Defining the samples
X = np.array([2,3,4,5], dtype=np.float32)
#f= w*x
y = np.array([4,6,8,10], dtype=np.float32)
# initializing the weigths
w = 0.0

#foward pass
def forward(x):
    return x*w

#loss
# For LR we use MSE as a loss function
def loss(y_hat, y):
    return ((y_hat-y)**2).mean()

#gradient
#MSE  = 1/N * (w*x-y)**2
#dJ/dw = 1/N * 2x * (w*x-y)

def gradient(x, y_hat, y):
    return np.dot(2*x, (y_hat-y)).mean()


print(f'prediction before training f(10)={forward(10)}')

#training 

#hyperparameters
learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    y_hat = forward(X)
    l = loss(y_hat, y)
    dw = gradient(X,y_hat, y)
    w -= learning_rate*dw

    if epoch % 1 ==0:
        print(f'epoch {epoch+1}: w = {w:0.3f}, loss  = {l:0.8f}')


print(f'prediction after training f(10)={forward(10)}')




