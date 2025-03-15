import torch
import torch.nn as nn
from metrics import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        
        # Values from the paper
        self.lambda_noObj = 0.5
        self.lambda_coord = 5


    def forward(self, predictions, target):
        # predictions are shaped (BTACH_SIZE, S*S(C+B*5)) whem inputed
        predictions = predictions.reshape(-1, self.S, self.C*self.B+5)

        # comppute the IoU for the predictions
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21,25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21,25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        # take the box with the highest IoU
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # This is the Iobj_j in the paper

        # compute the loss for the box coords

        '''
        Set the boxes with no obj in them to 0 
        we only take the box with one of the 2 predictions, we use the highest IoU compute earlyer
        '''
        box_predictions = exists_box*(
            (
                bestbox * predictions[..., 26:30]
                    +(1 - bestbox) * predictions[..., 21:25]
            )
        )
        box_targets = exists_box * target[..., 21:25]

        # take the sqrt to ensure that we dont favor bigger boxes
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1E-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # compute loss for the object
        # pred_box is the confidence score for the bbox with the highest IoU
        pred_box = (
            bestbox * predictions[..., 25:26] + (1-bestbox)* predictions[..., 20:21]
        )
        object_loss = self.mse(
            torch.mse(exists_box * pred_box),
            torch.mse(exists_box, target[..., 20:21])
        )

        # compute loss for no object
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )

        # each class loss
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        # final loss
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noObj * no_object_loss
            + class_loss
        )
        
        return loss
