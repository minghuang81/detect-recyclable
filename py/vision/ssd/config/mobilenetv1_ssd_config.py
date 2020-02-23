import numpy as np
import collections
import itertools
import math
import torch
import log

image_size = 300
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

SSDBoxSizes = collections.namedtuple('SSDBoxSizes', ['min', 'max'])
# feature_map_size : nb of grids that divide a image
# shrinkage : nb of pixels in a grid
# boxe size(min,max): 
#   min : ancor box size in pixels; max: next larger ancor box size
# aspect_ratios: each ratio in the list derives one w/h=ration ancor and one w/h=1/ratio ancor
SSDSpec = collections.namedtuple('SSDSpec', ['feature_map_size', 'shrinkage', 'box_sizes', 'aspect_ratios'])
# in the order of small to larger ancor boxes.
# 6 feature map scales, at each scale are generated 6 different ancor boxes per feature map grid: 
# [w=h=min],[w=h=sqrt min*max],[w=min*sqrt1,min/sqrt1],[w=min/sqrt1,min*sqrt1],[w=min*sqrt2,min/sqrt2],[w=min/sqrt2,min*sqrt2]
# where min, max = original min, max / image size of 300; x_center,y_center also relative ti image size; one grid square size w=h=1/scale
# For all grids at all scales, there are 1+2x2+3x3+5x5+10x10+19x19 = 500 grids
# Total number of ancor boxes = 500 x 6 = 3000
specs = [
    SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
    SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
    SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
    SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
    SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
    SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
]

# center_form_priors [[center_x, center_y, w, h]], values are all of range [0..1] relative to the image size
priors = [] 

def generate_ssd_priors(specs, image_size, clamp=True):
    """Generate SSD Prior Boxes.

    It returns the center, height and width of the priors. The values are relative to the image size
    Args:
        specs: SSDSpecs about the shapes of sizes of prior boxes. i.e.
            specs = [
                SSDSpec(19, 16, SSDBoxSizes(60, 105), [2, 3]),
                SSDSpec(10, 32, SSDBoxSizes(105, 150), [2, 3]),
                SSDSpec(5, 64, SSDBoxSizes(150, 195), [2, 3]),
                SSDSpec(3, 100, SSDBoxSizes(195, 240), [2, 3]),
                SSDSpec(2, 150, SSDBoxSizes(240, 285), [2, 3]),
                SSDSpec(1, 300, SSDBoxSizes(285, 330), [2, 3])
            ]
        image_size: image size.
        clamp: if true, clamp the values to make fall between [0.0, 1.0]
    Returns:
        priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    priors = []
    for spec in specs:
        scale = float(image_size) / spec.shrinkage # scale means number of (horizontal or vertical) grids
        for j, i in itertools.product(range(spec.feature_map_size), repeat=2):
            x_center = (i + 0.5) / scale # range [0..1] with 0 meaning complet left corner of image, 1 complete right
            y_center = (j + 0.5) / scale

            # small sized square box. w h x y are related to image size
            size = float(spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # big sized square box
            size = math.sqrt(spec.box_sizes.max * spec.box_sizes.min)
            h = w = size / image_size
            priors.append([
                x_center,
                y_center,
                w,
                h
            ])

            # change h/w ratio of the small sized box
            size = float(spec.box_sizes.min)
            h = w = size / image_size
            for ratio in spec.aspect_ratios:
                ratio = math.sqrt(ratio)
                priors.append([
                    x_center,
                    y_center,
                    w * ratio,
                    h / ratio
                ])
                priors.append([
                    x_center,
                    y_center,
                    w / ratio,
                    h * ratio
                ])

    priors = torch.tensor(priors)
    if clamp:
        torch.clamp(priors, 0.0, 1.0, out=priors)
    return priors

priors = generate_ssd_priors(specs, image_size)

# np.set_printoptions(threshold=np.inf,suppress=True)
# log.once('generate_ssd_priors={} - of shape {}'.format(np.round(priors.numpy(),6),priors.shape))
