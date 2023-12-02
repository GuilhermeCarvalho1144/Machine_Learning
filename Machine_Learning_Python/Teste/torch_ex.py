import torch
import torch.nn as nn

'''
Linear regression(LR) using torch

f = w*x

For this example we will keep it simple
the focus here is not learn LR
the foucs in on t learning pytorch
'''

# Defining the samples
X = torch.tensor([[2],[3],[4],[5]], dtype=torch.float32)
#f= w*x
y = torch.tensor([[4],[6],[8],[10]], dtype=torch.float32)
# initializing the weigths

X_test = torch.tensor([10], dtype=torch.float32)

input_shape = output_shape = X.shape[1]

model = nn.Linear(input_shape, output_shape)

print(f'prediction before training f(10)={model(X_test).item()}')

#training 

#hyperparameters
learning_rate = 0.01
epochs = 10

#defining loss and opt
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


for epoch in range(epochs):
    y_hat  = model(X)
    l = loss(y, y_hat)
    l.backward()
    
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 1 ==0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():0.3f}, loss  = {l:0.8f}')


print(f'prediction after training f(10)={model(X_test).item()}')




