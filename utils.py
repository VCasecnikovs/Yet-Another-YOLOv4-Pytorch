import torch
from torch import nn
import numpy as np
import cv2
from PIL import Image
from torchvision.ops import nms

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def get_img_with_bboxes(img, bboxes, resize=True, labels=None, confidences= None):
    c, h, w = img.shape
    
    bboxes_xyxy = bboxes.clone()
    bboxes_xyxy[:, :4] = xywh2xyxy(bboxes[:, :4])
    if resize:
        bboxes_xyxy[:,0] *= w
        bboxes_xyxy[:,1] *= h
        bboxes_xyxy[:,2] *= w
        bboxes_xyxy[:,3] *= h

        bboxes_xyxy[:, 0:4] = bboxes_xyxy[:,0:4].round()
    
    arr = bboxes_xyxy.numpy()

    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = (img * 255).astype(np.uint8) 
    
    #Otherwise cv2 rectangle will return UMat without paint
    img_ = img.copy()

    for i, bbox in enumerate(arr):
        img_ = cv2.rectangle(img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)  
        if labels:
            text = labels[i]
            text += f" {bbox[4].item() :.2f}"

            img_ = cv2.putText(img_, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255))
    return img_

def get_bboxes_from_anchors(anchors, confidence_threshold, iou_threshold, labels_dict):
    nbatches = anchors.shape[0]
    batch_bboxes = []
    labels = []

    for nbatch in range(nbatches):
        img_anchor = anchors[nbatch]
        confidence_filter = img_anchor[:, 4] > confidence_threshold
        img_anchor = img_anchor[confidence_filter]
        keep = nms(xywh2xyxy(img_anchor[:, :4]), img_anchor[:, 4], iou_threshold)
        img_bboxes = img_anchor[keep]
        batch_bboxes.append(img_bboxes)
        if len(img_bboxes) == 0:
            labels.append([])
            continue
        labels.append([labels_dict[x.item()] for x in img_bboxes[:, 5:].argmax(1)])

    return batch_bboxes, labels
     
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 +1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True, get_areas = False):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = (b1_area + b2_area - inter_area + 1e-16)


    if get_areas:
        return inter_area, union_area

    iou = inter_area / union_area
    return iou


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)  
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    target_boxes_grid = FloatTensor(nB, nA, nG, nG, 4).fill_(0)

    # 2 3 xy
    # 4 5 wh
    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    #Setting target boxes to big grid, it would be used to count loss
    target_boxes_grid[b, best_n, gj, gi] = target_boxes

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0


    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label (WE USE LABEL SMOOTHING)
    tcls[b, best_n, gj, gi, target_labels] = 0.9

    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()

    return iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes_grid



