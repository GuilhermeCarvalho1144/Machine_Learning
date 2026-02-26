import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchinfo

SEED = 42
# torch.use_deterministic_algorithms(True)


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH = 256
EPOCHS = 100
NUM_VAL = 10
model_output_path = "./ckpt/"

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, (2, 2))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, (2, 2))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def flatten(self, x):
        num_feats = 1
        for s in x.size()[1:]:
            num_featss *= s
        return num_feats


class AlexNet(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super(AlexNet, self).__init__(*args, **kwargs)
        self.feats = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=11,
                stride=4,
                padding=5,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=192, kernel_size=5, padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=192, out_channels=384, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=384, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fn = nn.Linear(in_features=256, out_features=4096)
        self.clf = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, input):
        x = self.feats(input)
        x = x.flatten(1)
        x = self.fn(x)
        x = self.clf(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes, *args, **kwargs) -> None:
        super(VGG, self).__init__(*args, **kwargs)
        self.feats = nn.Sequential(
            # (32x32x3) -> (32x32x64)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # (32x32x64) -> (32x32x64)
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (32x32x64) -> (16x16x64)
            nn.MaxPool2d(2, 2),
            # (16x16x64) -> (16x16x64)
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (16x16x64) -> (16x16x64)
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (16x16x64) -> (8x8x64)
            nn.MaxPool2d(2, 2),
            # (8x8x64) -> (8x8x128)
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (8x8x128) -> (8x8x128)
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (8x8x128) -> (8x8x128)
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (8x8x128) -> (4x4x128)
            nn.MaxPool2d(2, 2),
            # (4x4x128) -> (4x4x256)
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (4x4x256) -> (4x4x256)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (4x4x256) -> (4x4x256)
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            # (4x4x256) -> (4x4x512)
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (4x4x512) -> (4x4x512)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # (4x4x512) -> (4x4x512)
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
        )

    def forward(self, input):
        x = self.feats(input)
        x = self.avg(x)
        x = self.clf(x)
        return x


def data_loader():
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transforms
    )

    valid_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=test_transforms
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", download=True, train=False, transform=test_transforms
    )

    generator = torch.Generator().manual_seed(SEED)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size

    train_subset, _ = random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    _, valid_subset = random_split(
        valid_dataset, [train_size, val_size], generator=generator
    )

    workers = min(8, os.cpu_count()) if os.cpu_count() else 2

    train_dataloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=TRAIN_BATCH,
        shuffle=True,
        pin_memory=True,
        num_workers=workers,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_subset,
        batch_size=TRAIN_BATCH,
        shuffle=False,
        pin_memory=True,
        num_workers=workers,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10000, shuffle=False, pin_memory=True
    )

    return train_dataloader, valid_dataloader, test_dataloader


def train(model, dataloader, opt, loss_fn, epoch):
    loss_value = 0.0
    for n, data in enumerate(dataloader, 0):
        img, label = data
        img = img.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
        label = label.to(device, non_blocking=True)
        pred = model(img)
        loss = loss_fn(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_value += loss.item()
    print(f"Epoch: {epoch}: AVG loss: {loss_value/len(dataloader):.4f}")
    loss_value = 0.0


def validate(model, dataloader, loss_fn):
    model.eval()
    loss_value = 0.0

    sucess = 0
    count = 0

    with torch.no_grad():
        for img, label in dataloader:
            img, label = img.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            ), label.to(device, non_blocking=True)

            pred = model(img)
            loss = loss_fn(pred, label)

            loss_value += loss.item()

            _, pred_class = torch.max(pred, 1)
            count += label.size(0)
            sucess += (pred_class == label).sum().item()
    avg_loss = loss_value / len(dataloader)
    acc = 100 * (sucess / count)

    print(f"Validation AVG loss: {avg_loss:.4f}")
    print(f"Validation AVG ACC: {acc}")
    return acc


def test(model_output_path, dataloader):
    sucess = 0
    count = 0

    class_sucess = list(0.0 for _ in range(10))
    class_count = list(0.0 for _ in range(10))
    # model = LeNet()
    # model = AlexNet(len(classes))
    model = VGG(len(classes))
    model.load_state_dict(torch.load(model_output_path))
    with torch.no_grad():
        for data in dataloader:
            img, gt = data
            img.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
            gt.to(device, non_blocking=True)
            pred = model(img)
            _, pred_class = torch.max(pred, 1)
            count += gt.size(0)
            sucess += (pred_class == gt).sum().item()

            c = (pred_class == gt).squeeze()
            for i in range(10000):
                gt_current = gt[i]
                class_sucess[gt_current] += c[i].item()
                class_count[gt_current] += 1
    print(f"Model overall ACC: {(100*(sucess/count)):.2f}%")
    print("ACC per class")
    for i in range(10):
        print(
            f"ACC class {classes[i]}: {100*(class_sucess[i]/class_count[i]):.2f}%"
        )


if __name__ == "__main__":
    # model = LeNet()
    # model = AlexNet(len(classes))
    # torch.backends.cudnn.benchmark = True
    model = VGG(len(classes))
    model.to(device)
    torchinfo.summary(model, input_size=(1, 3, 32, 32))
    train_data, valid_data, test_data = data_loader()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    best_acc = 0.0
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        train(model, train_data, opt, loss_fn, epoch)
        if epoch % NUM_VAL == 0:
            current_acc = validate(model, valid_data, loss_fn)
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save(
                    model.state_dict(),
                    os.path.join(model_output_path, "best_ckpt.pb"),
                )
                print(f"New best model ACC: {current_acc}")
    torch.save(
        model.state_dict(), os.path.join(model_output_path, "last_ckpt.pb")
    )
    print("TESTING BEST CKPT")
    test(os.path.join(model_output_path, "best_ckpt.pb"), test_data)
    print("TESTING LAST CKPT")
    test(os.path.join(model_output_path, "last_ckpt.pb"), test_data)
    print("DONE")
