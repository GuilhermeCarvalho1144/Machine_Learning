import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


class AutoencoderMLP(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(28 * 28, 128),  # n, (28*28) -> n,512
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 18),
            nn.ReLU(),
            nn.Linear(18, 9),  # n, 32 -> n,4 (final dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(9, 18),  # n,4 -> n,32
            nn.ReLU(),
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, (28 * 28)),  # n,512 -> n,(28*28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoencoderMLP().to(device)
transform = transforms.ToTensor()
dataset = datasets.MNIST(
    root="../MNIST/data/", train=True, download=True, transform=transform
)
data_loader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=32, shuffle=True
)

# data = iter(data_loader)
# for data_ in data:
#     img, label = data_
#     print(torch.min(img), torch.max(img))
#     break
loss_fn = nn.MSELoss()
# loss_fn = nn.L1Loss()
opt = optim.Adam(model.parameters(), lr=1e-3)
all_loss = []
epochs = 10
for epoch in range(epochs):
    for img, _ in data_loader:
        img = img.view(-1, 28 * 28).to(device)
        y_pred = model(img)
        loss = loss_fn(y_pred, img)
        all_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"EPOCH {epoch} \t LOSS {loss.item():.4f}")

plt.figure()
plt.plot(all_loss)
plt.title("LOSS")
plt.ylabel("LOSS value")
plt.xlabel("iterations")
plt.savefig("loss_fn.png")


# model eval
model.eval()
eval_data_loader = iter(data_loader)
eval_data, labels = next(eval_data_loader)

eval_img = eval_data.view(-1, 28 * 28).to(device)
eval_pred = model(eval_img)

fig, ax = plt.subplots(2, 10, figsize=(12, 6))
for i in range(10):
    ax[0, i].imshow(
        eval_img[i].cpu().detach().numpy().reshape(28, 28), cmap="gray"
    )
    ax[0, i].axis("off")
    ax[0, i].set_title(f"label: {labels[i]}")

    ax[1, i].imshow(
        eval_pred[i].cpu().detach().numpy().reshape(28, 28), cmap="gray"
    )
    ax[1, i].axis("off")
    ax[1, i].set_title(f"label: {labels[i]}")

plt.suptitle("EVALUTATION")
plt.tight_layout()
plt.savefig("eval_images.png")
