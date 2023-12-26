import numpy as np 
import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format='corners'):

    print(f'using format {box_format}')
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1]-boxes_preds[..., 2:3]/2
        box1_y1 = boxes_preds[..., 1:2]-boxes_preds[..., 3:4]/2
        box1_x2 = boxes_preds[..., 0:1]+boxes_preds[..., 2:3]/2
        box1_y2 = boxes_preds[..., 1:2]+boxes_preds[..., 3:4]/2

        box2_x1 = boxes_labels[..., 0:1]-boxes_labels[..., 2:3]/2
        box2_y1 = boxes_labels[..., 1:2]-boxes_labels[..., 3:4]/2 
        box2_x2 = boxes_labels[..., 0:1]+boxes_labels[..., 2:3]/2 
        box2_y2 = boxes_labels[..., 1:2]+boxes_labels[..., 3:4]/2 


    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2] 
        box1_x2 = boxes_preds[..., 2:3] 
        box1_y2 = boxes_preds[..., 3:4]   

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2] 
        box2_x2 = boxes_labels[..., 2:3] 
        box2_y2 = boxes_labels[..., 3:4]   
    else:
        print(f'ERROR format: {format} not supported')
        return 0


    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    intersection = (x2-x1).clamp(0)*(y2-y1).clamp(0)

    box1_area = abs((box1_x2-box1_x1)*(box1_y2-box1_y1))
    box2_area = abs((box2_x2-box2_x1)*(box2_y2-box2_y1))

    return intersection/(box1_area+box2_area-intersection)

    

def non_max_supression(bboxes, iou_threshold, threshold, box_format="corners"):
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

if __name__ == "__main__":
    t1_box1 = torch.tensor([0.8, 0.1, 0.2, 0.2])
    t1_box2 = torch.tensor([0.9, 0.2, 0.2, 0.2])


    t2_box1 = torch.tensor([2, 2, 6, 6])
    t2_box2 = torch.tensor([4, 4, 7, 8])
    test1 = intersection_over_union(t1_box1, t1_box2, "midpoint") 
    test2 = intersection_over_union(t2_box1, t2_box2, "corners") 
    assert torch.tensor([1/7])-test1 < 0.001
    assert torch.tensor([4/24])-test2
