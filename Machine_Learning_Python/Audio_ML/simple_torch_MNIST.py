import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm


def download_mnist_dataset():
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    return train_data, validation_data


class create_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu(x)
        return logits


if __name__ == "__main__":
    train_data, _ = download_mnist_dataset()
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 5
    for epoch in tqdm(range(epochs)):
        loss_value = 0
        for batch, (X, y) in enumerate(train_dataloader):
            model.train()
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
        loss_value /= len(train_dataloader)
        print(f"Epoch: {epoch} | Loss: {loss_value:.4f}")
