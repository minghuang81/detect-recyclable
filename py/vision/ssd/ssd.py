import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F
import log

from ..utils import box_utils
from collections import namedtuple
GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #


class SSD(nn.Module):
    # def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
    #              extras: nn.ModuleList, classification_headers: nn.ModuleList,
    #              regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
    def __init__(self, num_classes, base_net, source_layer_indexes,
                 extras, classification_headers,
                 regression_headers, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        # debug trace:  source_layer_add_ons = ModuleList()
        #print('--------> source_layer_add_ons = {}'.format(self.source_layer_add_ons))
                
                                                   
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)
            
    # def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, x):
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                # log.once('added_layer = None')
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
            # log.info('confidence {}-shape {}, location {}-shape {}'.format(confidence,confidence.shape,location,location.shape))
            # confidence tensor([[[ 2.3434, -2.2576],
            #                     [ 2.7234, -3.0550],
            #                     [ 1.5128, -1.4758],
            #                     ...,
            #                     [ 1.8732, -1.8331],
            #                     [ 1.5814, -1.1892],
            #                     [ 1.4422, -1.2032]]])-shape torch.Size([1, 2166, 2]) <= feature map size 19x19
            # location   tensor([[[ 0.6972,  1.3065, -1.2949, -0.5907],
            #                     [-0.2067,  1.1810, -1.1765,  0.2641],
            #                     [ 0.2220,  1.0293, -1.4519,  0.4725],
            #                     ...,
            #                     [-0.0043, -0.5227, -0.3754, -1.3809],
            #                     [-0.7840, -0.3199, -1.7446, -0.2406],
            #                     [-0.6938, -0.6065, -0.7964, -1.0670]]])-shape torch.Size([1, 2166, 4])
            # confidence tensor([[[ 1.5930, -2.4059],
            #         [ 2.6398, -3.5865],
            #         [ 1.7649, -1.7079],
            #         ...,
            #         [ 1.2710, -1.1059],
            #         [ 1.7137, -1.7039],
            #         [ 1.8786, -1.4320]]])-shape torch.Size([1, 600, 2]) <= feature map size 10x10
            # location tensor([[[ 1.1163,  0.3411,  0.2025, -0.0421],
            #         [ 0.5245,  0.8978, -1.9887, -0.8022],
            #         [ 0.8107,  1.2731, -1.2033, -0.8545],
            #         ...,
            #         [ 0.0358, -0.2294,  0.1761, -0.3246],
            #         [ 0.1853,  0.2687,  0.1979,  0.1216],
            #         [-0.1177, -0.2304,  0.3786, -0.4077]]])-shape torch.Size([1, 600, 4])

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        # log.once('extras {}-shape {}'.format(self.extras,len(self.extras)))
        # extras ModuleList(
        # (0): Sequential( ...
        # )
        # ...
        # (4): Sequential( ...
        # )        
        # )-shape 4
        
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            # log.once('confidence {}-shape {}\n  location {}-shape {}'.format(confidence,confidence.shape,location,location.shape))
            # confidence tensor([[[ 1.3150, -0.7869],..[ 1.6980, -1.1988]]])-shape torch.Size([1, 150, 2])
            # location tensor([[[ 3.4466e-01,  2.0164e-01, -7.4911e-01, -6.3567e-01],..
            #                   [ 2.1677e-01, -1.6849e-01,  1.1663e-01, -5.5195e-01]]])-shape torch.Size([1, 150, 4])
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

            # confidence 0 shape=torch.Size([1, 2166, 2])   : 19x19x6 = 2166 ancors, 2 candidate classes (background 0, 1 my class)
            # location 0 shape=torch.Size([1, 2166, 4])
            # confidence 1 shape=torch.Size([1, 600, 2])    : 10x10x6 = 600 ancors
            # location 1 shape=torch.Size([1, 600, 4])
            # confidence 2 shape=torch.Size([1, 150, 2])    : 5x5x6   = 150 ancors
            # location 2 shape=torch.Size([1, 150, 4])
            # confidence 3 shape=torch.Size([1, 54, 2])     : 3x3x6   = 54 ancors
            # location 3 shape=torch.Size([1, 54, 4])
            # confidence 4 shape=torch.Size([1, 24, 2])     : 2x2x6   = 24 ancors
            # location 4 shape=torch.Size([1, 24, 4])
            # confidence 5 shape=torch.Size([1, 6, 2])      : 1x1x6   = 6 ancors
            # location 5 shape=torch.Size([1, 6, 4])

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        
        # log.once('confidences.shape {}, locations.shape {}'.format(confidences.shape, locations.shape))
        # confidences.shape torch.Size([1, 3000, 2]), locations.shape torch.Size([1, 3000, 4])

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            # 2166+600+150+54+24+6=3000: confidences(1,3000,2), locations(1,3000,4)
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        # log.info('compute_header confidence.shape  = {}'.format(confidence.shape))
        # 6 extras layers, 2 classes x 6 boxes = 12:
        # 0 compute_header confidence.shape  = torch.Size([1, 12, 19, 19])
        # 1 compute_header confidence.shape  = torch.Size([1, 12, 10, 10])
        # 2 compute_header confidence.shape  = torch.Size([1, 12, 5, 5])
        # 3 compute_header confidence.shape  = torch.Size([1, 12, 3, 3])
        # 4 compute_header confidence.shape  = torch.Size([1, 12, 2, 2])
        # 5 compute_header confidence.shape  = torch.Size([1, 12, 1, 1])

        confidence = confidence.permute(0, 2, 3, 1).contiguous() # torch.Size([1, 19, 19, 12])
        confidence = confidence.view(confidence.size(0), -1, self.num_classes) # torch.Size([1, 2166, 2])

        location = self.regression_headers[i](x)
        # log.info('{}: compute_header location.shape  = {}'.format(i,location.shape))
        # 6 extras layers, 6 boxes x 4 corners = 24 :
        # 0: compute_header location.shape  = torch.Size([1, 24, 19, 19])
        # 1: compute_header location.shape  = torch.Size([1, 24, 10, 10])
        # 2: compute_header location.shape  = torch.Size([1, 24, 5, 5])
        # 3: compute_header location.shape  = torch.Size([1, 24, 3, 3])
        # 4: compute_header location.shape  = torch.Size([1, 24, 2, 2])
        # 5: compute_header location.shape  = torch.Size([1, 24, 1, 1])
       
        location = location.permute(0, 2, 3, 1).contiguous() # torch.Size([1, 19, 19, 24])
        location = location.view(location.size(0), -1, 4) # torch.Size([1, 2166,4])

        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        # print('--------> state_dict = {}'.format(state_dict))
        model_dict = self.state_dict()
        #print('--------> model_dict  len {} = {}'.format(len(model_dict), dir(model_dict)))
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        #center_form_priors: the center, height and width of the priors. The values are relative to the image size
        self.center_form_priors = center_form_priors
        # corner_form_priors: also values related to the image size
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors) 
        # np.set_printoptions(threshold=np.inf,suppress=True)
        # log.once('corner_form_priors={} - of shape {}'.format(np.round(self.corner_form_priors.numpy(),6),self.corner_form_priors.shape))
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    # Argument
    #   gt_boxes : corner form from OpenImagesDataset._getitem() - [[xmin,ymin,xmax,ymax]] in unit of image size
    # return
    #   locations : center-form locations (3000,4) of gt_boxes "projected" onto the 3000 Priors.
    #               Only one of the gt_boxes is chosen for each Prior, then its location is revaluated.
    #               The center of a location is relative to the Prior's center then normalized to the Prior's size;
    #               the height/width of a location is normalized first to the Prior's size, then the log is taken.
    #   labels    : 3000 class id assigned to each of the 3000 Prior boxes. The class id is the one of the gt box 
    #               chosen for the Prior, it corresponds to the boxed object within the gt_box. 
    #   
    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        # log.once('gt_boxes={}, gt_labels={}'.format(gt_boxes,gt_labels))
        # gt_boxes(corner-form)=tensor([[0.0000, 0.1985, 0.3302, 0.5219]]), gt_labels=tensor([1])
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        # log.once('assigned boxes={}-shape {}, labels={}-shape {}, none-backgrounds={}'.format(boxes,boxes.shape,labels,labels.shape,(labels>0).sum()))  
        # log.once('assigned none-background labels={}'.format(labels[labels>0]))
        # assigned boxes(corner)=tensor([
        # [0.0000, 0.1985, 0.3302, 0.5219],
        # [0.0000, 0.1985, 0.3302, 0.5219],
        # [0.0000, 0.1985, 0.3302, 0.5219],
        # ...,
        # [0.0000, 0.1985, 0.3302, 0.5219],
        # [0.0000, 0.1985, 0.3302, 0.5219],
        # [0.0000, 0.1985, 0.3302, 0.5219]])-shape (3000,4), labels=tensor([0, 0, 0,  ..., 0, 0, 0])-shape (3000)
        boxes = box_utils.corner_form_to_center_form(boxes)
        # log.once('to center_form boxes={}'.format(boxes))
        #to center_form boxes=tensor([
        # [0.1651, 0.3602, 0.3302, 0.3234],
        # [0.1651, 0.3602, 0.3302, 0.3234],
        # [0.1651, 0.3602, 0.3302, 0.3234],
        # ...,
        # [0.1651, 0.3602, 0.3302, 0.3234],
        # [0.1651, 0.3602, 0.3302, 0.3234],
        # [0.1651, 0.3602, 0.3302, 0.3234]])-shape(3000,4)
        # location = (center_x,center_y relative to default box then normalized to that box size,
        #             log(w/default box w), log(h/default box h))
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        # log.once('locations of the center_form boxes={}'.format(locations))
        # locations of the center_form boxes=tensor([
        # [ 6.9213, 16.6752,  2.5067,  2.4026],
        # [ 5.2320, 12.6052,  1.1077,  1.0036],
        # [ 4.8941, 23.5822,  0.7738,  4.1355],
        # ...,
        # [-4.9856, -1.3983, -3.5512, -5.6446],
        # [-3.3491, -2.5494, -5.5405, -2.6416],
        # [-6.1061, -1.3983, -2.5375, -5.6446]]) -shape(3000,4)
        return locations, labels


# def _xavier_init_(m: nn.Module):
def _xavier_init_(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
