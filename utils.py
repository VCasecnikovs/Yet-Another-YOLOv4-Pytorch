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

        # get the coordinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0
        )
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = (b1_area + b2_area - inter_area + 1e-16)


        if get_areas:
            return inter_area, union_area

        iou = inter_area / union_area
        return iou

def nms_with_depth(bboxes, confidence, iou_threshold, depth_layer, depth_threshold):
    if len(bboxes) == 0:
        return bboxes

    for i in range(bboxes.shape[0]):
        for j in range(i+1, bboxes.shape[0]):
            iou = bbox_iou(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                #Getting center depth points of both bboxes
                D_oi = depth_layer[(bboxes[i, 0] + bboxes[i, 2])//2, (bboxes[i, 1] + bboxes[i, 3])//2]
                D_oj = depth_layer[(bboxes[j, 0] + bboxes[j, 2])//2, (bboxes[j, 1] + bboxes[j, 3])//2]
                if D_oi - D_oj < depth_threshold:
                    average_depth_oi = depth_layer[bboxes[i, 0] : bboxes[i, 2], bboxes[i, 1] : bboxes[i, 3]]
                    average_depth_oj = depth_layer[bboxes[j, 0] : bboxes[j, 2], bboxes[j, 1] : bboxes[j, 3]]
                    score_oi = confidence[i] + 1/torch.log(average_depth_oi)
                    score_oj = confidence[j] + 1/torch.log(average_depth_oj)
                    if score_oi > score_oj:
                        confidence[j] = 0
                    else:
                        confidence[i] = 0
    
    return confidence != 0

def matrix_nms(boxes, confidence, iou_threshold, batch_size, method, sigma, N):
    boxes = boxes.reshape(batch_size, -1)
    intersection = torch.mm(boxes, boxes.T)
    areas = boxes.sum(dim=1).expand(N, N)
    union = areas + areas.T - intersection
    ious = (intersection / union).triu(diagonal=1)

    ious_cmax = ious.max(0)
    ious_cmax = ious_cmax.expand(N, N).T

    if method == "gauss": # gaussian
        decay = torch.exp(-(ious**2 - ious_cmax**2) / sigma)
    else: # linear
        decay = (1 - ious) / (1 - ious_cmax)

    decay = decay.min(dim=0)
    return confidence * decay


def get_bboxes_from_anchors(anchors, confidence_threshold, iou_threshold, labels_dict, depth_layer = None, depth_threshold = 0.1):
    nbatches = anchors.shape[0]
    batch_bboxes = []
    labels = []

    for nbatch in range(nbatches):
        img_anchor = anchors[nbatch]
        confidence_filter = img_anchor[:, 4] > confidence_threshold
        img_anchor = img_anchor[confidence_filter]
        if depth_layer != None:
            keep = nms_with_depth(xywh2xyxy(img_anchor[:, :4]), img_anchor[:, 4], iou_threshold, depth_layer, depth_threshold)
        else:
            keep = nms(xywh2xyxy(img_anchor[:, :4]), img_anchor[:, 4], iou_threshold)
            
        img_bboxes = img_anchor[keep]
        batch_bboxes.append(img_bboxes)
        if len(img_bboxes) == 0:
            labels.append([])
            continue
        labels.append([labels_dict[x.item()] for x in img_bboxes[:, 5:].argmax(1)])

    return batch_bboxes, labels
     

def iou_all_to_all(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def smooth_ln(x, smooth =0.5):
    return torch.where(
        torch.le(x, smooth),
        -torch.log(1 - x),
        ((x - smooth) / (1 - smooth)) - np.log(1 - smooth)
    )

def iog(ground_truth, prediction):

    inter_xmin = torch.max(ground_truth[:, 0], prediction[:, 0])
    inter_ymin = torch.max(ground_truth[:, 1], prediction[:, 1])
    inter_xmax = torch.min(ground_truth[:, 2], prediction[:, 2])
    inter_ymax = torch.min(ground_truth[:, 3], prediction[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = (ground_truth[:, 2] - ground_truth[:, 0]) * (ground_truth[:, 3] - ground_truth[:, 1])
    return I / G
