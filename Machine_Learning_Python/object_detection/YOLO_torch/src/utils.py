import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import numpy as np 
from metrics import non_max_supression



def plot_img(img, boxes):
    #plot predict images
    image = np.array(img)
    heigth, width, _ = image.shape # heigth x width x n_chaneels 
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # boxes -> [x, y, width, heigth]
    for box in boxes:
        box = box[2:]
        assert len(box) == 4
        upper_left_x = (box[0] - box[2]) /2
        upper_left_y = (box[1] - box[3]) /2

        bbox = patches.Rectangle(
            (upper_left_x*width, upper_left_y*heigth),
            box[2]*width,
            box[3]*heigth,
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(bbox)
    plt.show()


def get_bboxes(
    loader, model, iou_threshold, threshold, 
    pred_format='cells', box_format='midponit', device='cuda'
    ):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(x)
        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(preds)

        for idx in range(batch_size):
            nms_boxes = non_max_supression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx]+nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx]+box)
            
            train_idx +=1
    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7):
    '''
    This code converts a image from YOLO pattern, SxS splits, to "normal" pattern
    It converts the points, width and heigth from a cell ration to a entire image ration
    This code is a vectorize imeplematation, very hard to read
    '''

    predictions = predictions.to('cpu')
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7,7,30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1-best_box) + best_box *bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1/S * (best_boxes[..., :1]+cell_indices)
    y = 1/S * (best_boxes[..., 1:2]+cell_indices.permute(0,2,1,3))
    w_y = 1/S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x,y,w_y), dim=-1)
    predicted_class = predictions[...,:20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(
        predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1)
    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_preds = convert_cellboxes(out).reshape(out.shape[0], S*S, -1)
    converted_preds[..., 0] = converted_preds[..., 0].long()
    all_boxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S*S):
            bboxes.append([x.item() for x in converted_preds[ex_idx, bbox_idx, :]])
        all_boxes.append(bboxes)
    return all_boxes

def save_checkpoint(state, filename):
    print('6*-> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print('6*-> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])














