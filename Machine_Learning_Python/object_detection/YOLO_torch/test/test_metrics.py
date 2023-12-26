from pytest import fixture, mark
from src import metrics as sut
import torch

eps = 0.1

@fixture
def test_1_midpoint():
    return torch.tensor([0.8, 0.1, 0.2, 0.2]) , torch.tensor([0.9, 0.2, 0.2, 0.2])  
@fixture
def test_2_center():
    return  torch.tensor([2, 2, 6, 6]), torch.tensor([4, 4, 7, 8]) 
@fixture
def test_1_boxes():
    return [[1, 1, 0.5, 0.45, 0.4, 0.5],
            [1, 0.8, 0.5, 0.5, 0.2, 0.4],
            [1, 0.7, 0.25, 0.35, 0.3, 0.1],
            [1, 0.05, 0.1, 0.1, 0.1, 0.1],]
@fixture
def test_1_boxes_result():
    return [[1, 1, 0.5, 0.45, 0.4, 0.5], [1, 0.7, 0.25, 0.35, 0.3, 0.1]]

@mark.iou
def test_tensor_1_iou_equals_1_over_7(test_1_midpoint):
    assert sut.intersection_over_union(test_1_midpoint[0], test_1_midpoint[1], 'midpoint')-(1/7) < eps

@mark.iou
def test_tensor_2_iou_equals_4_over_27(test_2_center):
    assert sut.intersection_over_union(test_2_center[0], test_2_center[1], 'center')-(4/27) < eps


def test_1_boxes_nms(test_1_boxes, test_1_boxes_result):
    result = sut.non_max_supression(test_1_boxes, threshold=0.2, iou_threshold=7/20, box_format='midpoint')
    assert sorted(result) == sorted(test_1_boxes_result)
