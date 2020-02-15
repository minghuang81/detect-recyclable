import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import log

class OpenImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None

    # output image : clipped, scaled training picture (300,300)
    # boxes, 
    # labels : (3000) class id 0 or 1 assigned to each default box (ancor). Only 2 classes i this log: object and background (0)
    def _getitem(self, index):
        image_info = self.data[index]
        # sub-train-annotations-bbox.csv:
        #   ImageID,Source,LabelName,Confidence, XMin, XMax,    YMin,    YMax,              IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,id,ClassName
        #   6f559d2815699edf,xclick,/m/0hkxq,1,0.14875,0.279375,0.608939,0.8212290000000001,0,0,0,0,0,/m/0hkxq,Broccoli
        #   class id /m/0hkxq = Broccoli, jpg image size: w1024 h573, bbox normalized to image size
        #log.once(image_info)
        #   {'image_id': '6f559d2815699edf', 'boxes': array([[0.14875 , 0.608939, 0.279375, 0.821229]], dtype=float32), 'labels': array([1])
        #   boxes = [[xmin,ymin,xmax,ymax]] normalized to image width, hight
        image = self._read_image(image_info['image_id'])
        # image.shape (h573, w1024, c3)
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])

        if (boxes.max() < 1.01):    # box coord. are normalized to the image size
            boxes[:, 0] *= image.shape[1] # image w 1024
            boxes[:, 1] *= image.shape[0] # image h 573
            boxes[:, 2] *= image.shape[1] # image w
            boxes[:, 3] *= image.shape[0] # image h
            log.once('box coord. are normalized to the image size')
        else:
            log.once('box coord. are expressed in pixels')

        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        #log.once('bef transform image.shape {},boxes {},labels {}'.format(image.shape, boxes, labels))
        # bef transform image.shape (h573, w1024, c3),boxes [[152.32 348.92203 286.08 470.5642 ]],labels [1]
        # boxes = [[xmin,ymin,xmax,ymax]] in pixles
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        #log.once('aft transform image.shape {},boxes {},labels {}'.format(image.shape, boxes, labels))
        # aft transform image.shape torch.Size([3, 300, 300]),boxes [[0.19776616 0.6475796  0.43120417 0.8779625 ]],labels [1]
        # boxes = [[xmin,ymin,xmax,ymax]] in unit of image size
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        #log.once('aft traget tr image.shape {},boxes {}-{} items,labels {}-{} items'.format(image.shape, boxes, boxes.shape[0],labels,labels.shape[0]))
        # aft traget tr image.shape torch.Size([3, 300, 300]),
        # boxes tensor([[14.3909, 36.8052,  0.7730,  0.7071],
        #               [10.8785, 27.8221, -0.6260, -0.6919],
        #               [10.1759, 52.0504, -0.9599,  2.4400],
        #               ...,
        #               [-2.7617,  2.6277, -5.2849, -7.3401],
        #               [-1.8551,  4.7909, -7.2742, -4.3371],
        #               [-3.3823,  2.6277, -4.2712, -7.3401]])-3000 items,labels tensor([0, 0, 0,  ..., 0, 0, 0])-3000 items
        # box = (center_x rel. to ancor box center, center_y rel.to ancor box, log(w/ancor box w), log(h/ancor box h))
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = str(self.root)+"/sub-"+self.dataset_type+"-annotations-bbox.csv"
        annotations = pd.read_csv(annotation_file)
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # make labels 64 bits to satisfy the cross_entropy function
            labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   "Number of Images: {} "
                   "Minimum Number of Images for a Class: {} "
                   "Label Distribution:"
                   .format(len(self.data),self.min_image_num)]
        for class_name, num in self.class_stat.items():
            content.append("\t{}: {}".format(class_name, num))
        return "\n".join(content)

    def _read_image(self, image_id):
        # image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        # image_file = "{} / {} / {}.jpq".format(self.root, self.dataset_type, image_id)
        image_file = self.root / self.dataset_type / (str(image_id)+".jpg")
        # print("image_file = ", image_file)
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data





