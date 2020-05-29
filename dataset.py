import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
import utils
import random


class ListDataset(Dataset):
    def __init__(self, list_path, img_dir = "images", labels_dir="labels",  img_extensions=[".JPG"], img_size=608, train=True, bbox_minsize = 0.01, brightness_range=0.25, contrast_range=0.25, hue_range=0.05, saturation_range=0.25, cross_offset = 0.2):
        with open(list_path, "r") as file:
            self.img_files = file.read().splitlines()


        self.label_files = []
        for path in self.img_files:
            path = path.replace(img_dir, labels_dir)
            for ext in img_extensions:
                path = path.replace(ext, ".txt")
            self.label_files.append(path)

        self.img_size = img_size
        self.to_tensor =  transforms.ToTensor()

        self.train = train

        self.bbox_minsize = bbox_minsize

        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range

        self.cross_offset = cross_offset

    def __getitem__(self, index):

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()


        # Getting image
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        #RESIZING
        if width > height:
            ratio = height/width
            t_width = self.img_size
            t_height = int(ratio * self.img_size) 
            
        else:
            ratio = width/height
            t_width = int(ratio * self.img_size)
            t_height = self.img_size

        img = transforms.functional.resize(img, (t_height, t_width))
        
        #IF TRAIN APPLY BRIGHTNESS CONTRAST HUE SATURTATION
        if self.train:
            brightness_rnd = random.uniform(1- self.brightness_range, 1 + self.brightness_range)
            contrast_rnd = random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            hue_rnd = random.uniform(-self.hue_range, self.hue_range)
            saturation_rnd = random.uniform(1 - self.saturation_range, 1 + self.saturation_range)

            img = transforms.functional.adjust_brightness(img, brightness_rnd)
            img = transforms.functional.adjust_contrast(img, contrast_rnd)
            img = transforms.functional.adjust_hue(img, hue_rnd)
            img = transforms.functional.adjust_saturation(img, saturation_rnd)


        #CONVERTING TO TENSOR
        tensor_img = transforms.functional.to_tensor(img)

        # Handle grayscaled images
        if len(tensor_img.shape) != 3:
            tensor_img = tensor_img.unsqueeze(0)
            tensor_img = tensor_img.expand((3, img.shape[1:]))



        #!!!WARNING IN PIL IT'S WIDTH HEIGHT, WHEN IN PYTORCH IT IS HEIGHT WIDTH

        # Apply augmentations for train it would be mosaic
        if self.train:
            mossaic_img = torch.zeros(3, self.img_size, self.img_size)

            #FINDING CROSS POINT
            cross_x = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))
            cross_y = int(random.uniform(self.img_size * self.cross_offset, self.img_size * (1 - self.cross_offset)))

            fragment_img, fragment_bbox = self.get_mosaic(0, cross_x, cross_y, tensor_img, boxes)
            mossaic_img[:, 0:cross_y, 0:cross_x] = fragment_img
            boxes = fragment_bbox
            

            for n in range(1, 4):
                raw_fragment_img, raw_fragment_bbox = self.get_img_for_mosaic(brightness_rnd, contrast_rnd, hue_rnd, saturation_rnd)
                fragment_img, fragment_bbox = self.get_mosaic(n, cross_x, cross_y, raw_fragment_img, raw_fragment_bbox)
                boxes = torch.cat([boxes, fragment_bbox])

                if n == 1:
                    mossaic_img[:, 0 : cross_y, cross_x : self.img_size] = fragment_img
                elif n == 2:
                    mossaic_img[:, cross_y : self.img_size, 0 : cross_x] = fragment_img
                elif n == 3:
                    mossaic_img[:, cross_y : self.img_size, cross_x : self.img_size] = fragment_img

            #Set mossaic to return tensor
            tensor_img = mossaic_img


        # For validation it would be letterbox
        else:
            xyxy_bboxes = utils.xywh2xyxy(boxes[:, 1:])

            #IMG
            padding = abs((t_width - t_height))//2
            padded_img = torch.zeros(3, self.img_size, self.img_size)
            if t_width > t_height:
                padded_img[:, padding:padding+t_height] = tensor_img
            else:
                padded_img[:, :, padding:padding+t_width] = tensor_img

            tensor_img = padded_img
            
            relative_padding = padding/self.img_size
            #BOXES
            if t_width > t_height:
                #Change y's relative position
                xyxy_bboxes[:, 1] *= ratio
                xyxy_bboxes[:, 3] *= ratio
                xyxy_bboxes[:, 1] += relative_padding
                xyxy_bboxes[:, 3] += relative_padding
            else:#x's
                xyxy_bboxes[:, 0] *= ratio
                xyxy_bboxes[:, 2] *= ratio
                xyxy_bboxes[:, 0] += relative_padding
                xyxy_bboxes[:, 2] += relative_padding

            boxes[:, 1:] = utils.xyxy2xywh(xyxy_bboxes)


        
        
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes


        return img_path, tensor_img, targets

    def get_img_for_mosaic(self, brightness_rnd, contrast_rnd, hue_rnd, saturation_rnd):
        random_index = random.randrange(0, len(self.img_files))
        img_path = self.img_files[random_index].rstrip()
        label_path = self.label_files[random_index].rstrip()

        

        # Getting image
        img = Image.open(img_path).convert('RGB')
        width, height = img.size

        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        #RESIZING
        if width > height:
            ratio = height/width
            t_width = self.img_size
            t_height = int(ratio * self.img_size) 
            
        else:
            ratio = width/height
            t_width = int(ratio * self.img_size)
            t_height = self.img_size

        img = transforms.functional.resize(img, (t_height, t_width))

        img = transforms.functional.adjust_brightness(img, brightness_rnd)
        img = transforms.functional.adjust_contrast(img, contrast_rnd)
        img = transforms.functional.adjust_hue(img, hue_rnd)
        img = transforms.functional.adjust_saturation(img, saturation_rnd)

        #CONVERTING TO TENSOR
        tensor_img = transforms.functional.to_tensor(img)

        # Handle grayscaled images
        if len(tensor_img.shape) != 3:
            tensor_img = tensor_img.unsqueeze(0)
            tensor_img = tensor_img.expand((3, img.shape[1:]))

        return tensor_img, boxes


    # N is spatial parameter if 0 TOP LEFT, if 1 TOP RIGHT, if 2 BOTTOM LEFT, if 3 BOTTOM RIGHT
    def get_mosaic(self, n, cross_x, cross_y, tensor_img, boxes):
        t_height = tensor_img.shape[1]
        t_width = tensor_img.shape[2]

        xyxy_bboxes = utils.xywh2xyxy(boxes[:, 1:])

        relative_cross_x = cross_x / self.img_size
        relative_cross_y = cross_y / self.img_size

        #CALCULATING TARGET WIDTH AND HEIGHT OF PICTURE
        if n == 0:
            width_of_nth_pic = cross_x 
            height_of_nth_pic = cross_y
        elif n == 1:
            width_of_nth_pic = self.img_size - cross_x
            height_of_nth_pic = cross_y
        elif n == 2:
            width_of_nth_pic = cross_x
            height_of_nth_pic = self.img_size - cross_y
        elif n == 3:
            width_of_nth_pic = self.img_size - cross_x
            height_of_nth_pic = self.img_size - cross_y

        #self.img_size - width_of_1st_pic
        #selg.img_size - height_of_1st_pic  


        #CHOOSING TOP LEFT CORNER (doing offset to have more than fex pixels in bbox :-) )
        cut_x1 = random.randint(0, int(t_width * 0.33))
        cut_y1 = random.randint(0, int(t_height * 0.33))


        #Now we should find which axis should we randomly enlarge (this we do by finding out which ratio is bigger); cross x is basically width of the top left picture
        if (t_width - cut_x1) / width_of_nth_pic < (t_height - cut_y1) / height_of_nth_pic:
            cut_x2 = random.randint(cut_x1 + int(t_width * 0.67), t_width)
            cut_y2 = int(cut_y1 + (cut_x2-cut_x1)/width_of_nth_pic*height_of_nth_pic)

        else:
            cut_y2 = random.randint(cut_y1 + int(t_height * 0.67), t_height)
            cut_x2 = int(cut_x1 + (cut_y2-cut_y1)/height_of_nth_pic*width_of_nth_pic)
        
        #RESIZING AND INSERTING (TO DO 2D interpolation wants 4 dimensions, so I add and remove one by using None and squeeze)
        tensor_img = F.interpolate(tensor_img[:, cut_y1:cut_y2,  cut_x1:cut_x2][None], (height_of_nth_pic, width_of_nth_pic)).squeeze()
        
        #BBOX
        relative_cut_x1 = cut_x1 / t_width
        relative_cut_y1 = cut_y1 / t_height
        relative_cropped_width = (cut_x2 - cut_x1) / t_width
        relative_cropped_height = (cut_y2 - cut_y1) / t_height

        #SHIFTING TO CUTTED IMG SO X1 Y1 WILL 0
        xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] - relative_cut_x1
        xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] - relative_cut_y1
        xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] - relative_cut_x1
        xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] - relative_cut_y1

        #RESIZING TO CUTTED IMG SO X2 WILL BE 1
        xyxy_bboxes[:, 0] /= relative_cropped_width
        xyxy_bboxes[:, 1] /= relative_cropped_height
        xyxy_bboxes[:, 2] /= relative_cropped_width
        xyxy_bboxes[:, 3] /= relative_cropped_height

        #CLAMPING BOUNDING BOXES, SO THEY DO NOT OVERCOME OUTSIDE THE IMAGE
        xyxy_bboxes[:, 0].clamp_(0, 1)
        xyxy_bboxes[:, 1].clamp_(0, 1)
        xyxy_bboxes[:, 2].clamp_(0, 1)
        xyxy_bboxes[:, 3].clamp_(0, 1)

        #FILTER TO THROUGH OUT ALL SMALL BBOXES
        filter_minbbox = (xyxy_bboxes[:, 2] - xyxy_bboxes[:, 0] > self.bbox_minsize) & (xyxy_bboxes[:, 3] - xyxy_bboxes[:, 1] > self.bbox_minsize)

        # RESIZING TO MOSAIC
        if n == 0:
            xyxy_bboxes[:, 0] *= relative_cross_x #
            xyxy_bboxes[:, 1] *= relative_cross_y #(1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= relative_cross_x #
            xyxy_bboxes[:, 3] *= relative_cross_y #(1 - relative_cross_y)
        elif n==1:
            xyxy_bboxes[:, 0] *= (1 - relative_cross_x) 
            xyxy_bboxes[:, 1] *= relative_cross_y
            xyxy_bboxes[:, 2] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 3] *= relative_cross_y
        elif n==2:
            xyxy_bboxes[:, 0] *= relative_cross_x
            xyxy_bboxes[:, 1] *= (1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= relative_cross_x
            xyxy_bboxes[:, 3] *= (1 - relative_cross_y)
        elif n==3:
            xyxy_bboxes[:, 0] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 1] *= (1 - relative_cross_y)
            xyxy_bboxes[:, 2] *= (1 - relative_cross_x)
            xyxy_bboxes[:, 3] *= (1 - relative_cross_y)

        #RESIZING TO MOSAIC
        if n == 0:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] #+ relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] #+ relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] #+ relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] #+ relative_cross_y
        elif n==1:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] + relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1]
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] + relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3]
        elif n==2:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0]
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] + relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2]
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] + relative_cross_y
        elif n==3:
            xyxy_bboxes[:, 0] = xyxy_bboxes[:, 0] + relative_cross_x
            xyxy_bboxes[:, 1] = xyxy_bboxes[:, 1] + relative_cross_y
            xyxy_bboxes[:, 2] = xyxy_bboxes[:, 2] + relative_cross_x
            xyxy_bboxes[:, 3] = xyxy_bboxes[:, 3] + relative_cross_y
        
        boxes = boxes[filter_minbbox]
        boxes[:, 1:] = utils.xyxy2xywh(xyxy_bboxes)[filter_minbbox]

        return tensor_img, boxes


    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)

        return paths, torch.stack(imgs), targets


    def __len__(self):
        return len(self.img_files)