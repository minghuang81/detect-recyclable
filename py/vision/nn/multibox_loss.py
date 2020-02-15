import torch.nn as nn
import torch.nn.functional as F
import torch
import log


from ..utils import box_utils


class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio      # neg_pos_ratio=3
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    # input confidence, predicted_locations : result of prediction
    # input labels, gt_locations : training dataset
    # return : 
    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding gt_boxes mapped to all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            # log_softmax(x) = log(softmax(x))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]   # loss for being classified as background 
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # log.once('confidence.shape {}, loss.shape {}, mask.shape {}'.format(confidence.shape,loss.shape,mask.shape))
        # confidence.shape torch.Size([5, 3000, 2]), loss.shape torch.Size([5, 3000]), mask.shape torch.Size([5, 3000])
        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)   # shape(X,4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)                 # shape(X,4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
