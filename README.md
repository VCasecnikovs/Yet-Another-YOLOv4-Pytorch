# Yet-Another-YOLOv4-Pytorch

This is implementation of YOLOv4 object detection neural network on pytorch. I'll try to implement all features of original paper.

 - [x] Model
 - [x] Pretrained weights
 - [x] Custom classes
 - [x] CIoU
 - [x] YOLO dataset
 - [x] Letterbox for validation
 - [x] HSV transforms for train
 - [x] MOSAIC for train
 - [ ] MiWRC for forward attention
 - [ ] Self attention attack
 - [ ] Notebook with guide

## Initialize NN

    import model
    #If you change n_classes from the pretrained, there will be caught one error, don't panic it is ok
    m = model.YOLOv4(n_classes=1, weights_path="weights/yolov4.pth")
    
## Initialize dataset

    import dataset
    d = dataset.ListDataset("train.txt", img_dir='images', labels_dir='labels', img_extensions=['.JPG'], train=True)
	path, img, bboxes = d[0]

"train.txt" is file which consists filepaths to image (images\primula\DSC02542.JPG)

img_dir - Folder with images
labels_dir - Folder with txt files for annotion
img_extensions - extensions if images

If you set train=False -> uses letterboxes
If you set train=True -> HSV augmentations and mosaic

dataset has collate_function

    # collate func example
    y1 = d[0]
    y2 = d[1]
    paths_b, xb, yb = d.collate_fn((y1, y2))
	# yb has 6 columns
	
### Bboxes format
 1. Num of img to which this anchor belongs
 2. BBox class
 3. x center
 4. y center
 5. width
 6. height
   
### Forward with loss
    (y_hat1, y_hat2, y_hat3), (loss_1, loss_2, loss_3) = m(xb, yb)

### Forward without loss
    (y_hat1, y_hat2, y_hat3), _ = m(img_batch) #_ is (0, 0, 0)

### Check if bboxes are correct
    import utils
    path, img, bboxes = d[0]
    img_with_bboxes = utils.get_img_with_bboxes(img, bboxes) #PIL image
	