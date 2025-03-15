import torch
import torch.nn as nn
import torch.optim as optim

# define vector
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)


# define MLP
class MLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.xor_model = nn.Sequential(
            nn.Linear(2, 6), nn.Sigmoid(), nn.Linear(6, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.xor_model(x)


# create ANN
model = MLP()
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters())

epochs = 10000
for epoch in range(epochs):
    out = model(x)
    loss = loss_fn(out, y)

    print(f"LOSS: {loss:.2}\t epoch: {epoch}")
    # update grads
    opt.zero_grad()
    loss.backward()
    opt.step()
