import torch
from model import YOLOv4

dependencies = ['torch']

def yolov4(pretrained=False, n_classes=80):
    """
    YOLOv4 model
    pretrained (bool): kwargs, load pretrained weights into the model
    n_classes(int): amount of classes
    """
    m = YOLOv4(n_classes=n_classes)
    if pretrained:
        try: #If we change input or output layers amount, we will have an option to use pretrained weights
            m.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/VCasecnikovs/Yet-Another-YOLOv4-Pytorch/releases/download/V1.0/yolov4.pth"), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')

    return m
    
           