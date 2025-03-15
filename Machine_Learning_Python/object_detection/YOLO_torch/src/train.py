from logging import shutdown
import torch
from torch.fx.node import Target
from torch.nn.modules import loss 
import torchvision.transforms as transforms
import torch .optim as optim
import torchvision.transforms.functional as FT 
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import YoloV1
from dataset import VOCDataset
from loss import YoloLoss
from metrics import (
    non_max_supression,
    intersection_over_union,
    mean_avarage_precision
)
from utils import (
    get_bboxes,
    plot_img,
    save_checkpoint,
    load_checkpoint
)

#seed
seed = 42
torch.manual_seed(seed)

#hyperparameters
LR = 1E-3 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
WEIGTH_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = 'save_model.pth.tar'
IMG_DIR = '../../../../../../Database/PascalVOC_YOLO/images/'
LABEL_DIR = '../../../../../../Database/PascalVOC_YOLO/labels/'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optmizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x)
        loss = loss_fn(preds, y)
        mean_loss.append(loss.item())
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        
        #update progress bar 
        loop.set_postfix(loss=loss.item())
    
    print(f'MEAN LOSS WAS {sum(mean_loss)/len(mean_loss)}')

def train():
    model = YoloV1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGTH_DECAY
    )
    loss_fn = YoloLoss()
    if LOAD_MODEL:
        load_checkpoint(
            torch.load(LOAD_MODEL_FILE),
            model,
            optimizer
        )


    train_dataset = VOCDataset(
        '../../../../../../Database/PascalVOC_YOLO/8examples.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )
    test_dataset = VOCDataset(
        '../../../../../../Database/PascalVOC_YOLO/test.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
        
    # training loop
    for epoch in range(EPOCHS):
        pred_boxes, target_box = get_bboxes(
            train_loader,
            model,
            iou_threshold=0.5,
            threshold=0.4
        )
        mean_avg_precision = mean_avarage_precision(
            pred_boxes,
            target_box,
            iou_threshold=0.5,
            box_format='midpoint'
        )

        train_fn(
            train_loader,
            model,
            optimizer,
            loss_fn
        )

if __name__ == "__main__":
    train()

