import collections
import torch
import itertools
from typing import List
import math
import log

from vision.ssd.config.mobilenetv1_ssd_config import specs, center_variance, size_variance, image_size, SSDSpec, SSDBoxSizes, priors
# SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
# # shrinkage : nb of pixels in a grid
# # feature_map_size : nb of grids that divide a image
# SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])

# specs = [
#     SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
#     SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
#     SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
# ]

# def generate_ssd_priors(specs: List[SSDSpec], image_size, clamp=True) -> torch.Tensor:
# def generate_ssd_priors(specs, image_size, clamp=True):
#     """Generate SSD Prior Boxes.

#     It returns the center, height and width of the priors. The values are relative to the image size
#     Args:
#         specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
#             specs = [
#                 SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#                 SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#                 SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#                 SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#                 SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#                 SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
#             ]
#         image_size: image size.
#         clamp: if true, clamp the values to make fall between [0.0, 1.0]
#     Returns:
#         priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
#             are relative to the image size.
#     """
#     priors = []
#     for spec in specs:
#         scale = image_size / spec.shrinkage # scale means number of (horizontal or vertical) grids
#         for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
#             x_center = (i + 0.5) / scale # range [0..1] with 0 meaning complet left corner of image, 1 complete right
#             y_center = (j + 0.5) / scale

#             # small sized square box. w h x y are related to image size
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])

#             # big sized square box
#             size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
#             h = w = size / image_size
#             priors.append([
#                 x_center,
#                 y_center,
#                 w,
#                 h
#             ])

#             # change h/w ratio of the small sized box
#             size = spec.box_sizes.min
#             h = w = size / image_size
#             for ratio in spec.aspect_ratios:
#                 ratio = math.sqrt(ratio)
#                 priors.append([
#                     x_center,
#                     y_center,
#                     w * ratio,
#                     h / ratio
#                 ])
#                 priors.append([
#                     x_center,
#                     y_center,
#                     w / ratio,
#                     h * ratio
#                 ])

#     priors = torch.tensor(priors)
#     if clamp:
#         torch.clamp(priors, 0.0, 1.0, out=priors)
#     return priors


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, h, w]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


# def area_of(left_top, right_bottom) -> torch.Tensor:
def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 shape(1, M, 4): M ground truth boxes.
        boxes1 (3000,1,4): 3000 predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values, of shape (3000,M)
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes corner form, values relative to image size.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors, values relative to image size.
    Returns:
        boxes (num_priors, 4): all output boxes ([3000,4]) are one of the gt_boxes, in corner-form and normalized to image size.
                               For each of the 3000 priors (ancor), the closest gt_boxes is chosen.
                               In the simplest case where there is only one ground-truth box drawn in the training image,
                               then the same gt box is repeated 3000 times in the returned boxes([3000,4])
        labels (num_priors)  : labels for priors - shape([3000]), 3000 labels assigned one to each prior (ancor).
                               The closest prior to a gt box is always assigned that gt box, so all gt_box are assigned at leat once each,
                               meaning you find all input gt_boxes in the output boxes.
                               All priors are assigned a gt box that is the closest, so in the output boxes, 
                               you see repetition of gt_boxes up to 'num_priors' times (3000).
                               However, most of output boxes are labeled with the background class 0. Only if a Prior overlaps
                               sufficiently with a gt_box, the corresponding output label is assigned the object's class ID instead of 0.
    """
    # ious.shape: (num_priors, num_targets) where num_priors=3000
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # log.once("ious: {} - shape {},gt shape {},priors shape {}".format(ious,ious.shape,gt_boxes.shape,corner_form_priors.shape))
    # log.once("gt_boxes.unsqueeze(0) shape {},corner_form_priors.unsqueeze(1) shape {}".format(gt_boxes.unsqueeze(0).shape,corner_form_priors.unsqueeze(1).shape))
    # ious: tensor([[0.0000],
    #     [0.0000],
    #     [0.0000],
    #     ...,
    #     [0.0855],
    #     [0.1216],
    #     [0.0673]]) - shape torch.Size([3000, 1]),gt shape torch.Size([1, 4]),priors shape torch.Size([3000, 4])
    # gt_boxes.unsqueeze(0)=shape torch.Size([1, 1, 4]),corner_form_priors.unsqueeze(1)=shape torch.Size([3000, 1, 4])
    # best_target_per_prior, best_target_per_prior_index (target index or class id): torch.Size([3000])
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # log.once("best_target_per_prior: {} - shape {}".format(best_target_per_prior,best_target_per_prior.shape))
    # best_target_per_prior: tensor([0.0000, 0.0000, 0.0000,  ..., 0.0855, 0.1216, 0.0673]) - shape torch.Size([3000])
    # log.once("best_target_per_prior_index: {} - shape {}".format(best_target_per_prior_index,best_target_per_prior_index.shape))
    # best_target_per_prior_index: tensor([0, 0, 0,  ..., 0, 0, 0]) - shape torch.Size([3000])
    # best_prior_per_target, best_prior_per_target_index shape: torch.Size([num_targets])
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
        # log.once("best_target_per_prior_index[{}] = {}".format(prior_index,target_index))
        # best_target_per_prior_index[2532] = 0
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # log.once("best_target_per_prior aft index_fill_: {} - shape {}".format(best_target_per_prior,best_target_per_prior.shape))
    # best_target_per_prior aft index_fill_: tensor([0.0000, 0.0000, 0.0000,  ..., 0.0855, 0.1216, 0.0673]) - shape torch.Size([3000])
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # 0 is the class id of the backgournd 
    # log.once("labels {} - shape {}, none backgrounds {}".format(labels,labels.shape, sum(labels>0)))
    # labels tensor([0, 0, 0,  ..., 0, 0, 0]) - shape torch.Size([3000]), none backgrounds 14
    # best_target_per_prior_index of shape([3000]),gt_boxes of shape([2,4]), gt_boxes[best_target_per_prior_index] = shape([3000,4])
    boxes = gt_boxes[best_target_per_prior_index] 
    # log.once("gt box assigned to each ancor box(prior): {} - shape {}".format(boxes,boxes.shape))
    # gt box assigned to each ancor box(prior): tensor(
    #    [[0.0000, 0.5708, 0.3539, 0.8976],
    #     [0.0000, 0.5708, 0.3539, 0.8976],
    #     [0.0000, 0.5708, 0.3539, 0.8976],
    #     ...,
    #     [0.0000, 0.5708, 0.3539, 0.8976],
    #     [0.0000, 0.5708, 0.3539, 0.8976],
    #     [0.0000, 0.5708, 0.3539, 0.8976]]) - shape torch.Size([3000, 4])
    return boxes, labels


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)  # number of positive pred within the same batch
    num_neg = num_pos * neg_pos_ratio

    # loss[pos_mask] = -math.inf
    loss[pos_mask] = -float('inf')
    _, indexes = loss.sort(dim=1, descending=True)  # indexes in descending order of loss, positives pushed to the end.
    _, orders = indexes.sort(dim=1)                 
    neg_mask = orders < num_neg                     # num_neg indexes having the highest loss[i] among all loss[:, 0:3000]

    # log.info('positive predictions / neg predictionsin {}/{}'.format(pos_mask.sum(),neg_mask.sum()))
    # positive predictions / neg predictionsin 22/66
    # positive predictions / neg predictionsin 49/147
    # positive predictions / neg predictionsin 93/279

    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:]/2.0,
                     locations[..., :2] + locations[..., 2:]/2.0], locations.dim() - 1) 


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2.0,
         boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]      # box_scores[:,4]
    boxes = box_scores[:, :-1]      # box_scores[:,0-1-2-3]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())   # current = tensor(0) whereas current.item() = 0 => picked = [0], not [tensor(0)]
        # log.once('current {}-shape {}, picked {}-len {}'.format(current,current.shape,picked,len(picked)))
        # current 0-shape torch.Size([]), picked [0]-len 1
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


# input box_scores(N,5) : boxes in corner-form and probabilities
# return: a subset of box_scores without change within a box (center-form remains center-form, size not altered)
def nms(box_scores, nms_method=None, score_threshold=None, iou_threshold=None,
        sigma=0.5, top_k=-1, candidate_size=200):
    if nms_method == "soft":
        return soft_nms(box_scores, score_threshold, sigma, top_k)
    else:
        return hard_nms(box_scores, iou_threshold, top_k, candidate_size=candidate_size)


def soft_nms(box_scores, score_threshold, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, 4])
        cur_box_prob = torch.tensor(box_scores[max_score_index, :])
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-1]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-1])
        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)
        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])



