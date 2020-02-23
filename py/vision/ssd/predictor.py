import torch
import log
from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer


class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method

        self.sigma = sigma
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net.to(self.device)
        self.net.eval()

        self.timer = Timer()

    # Return boxes, labels, probs:
    # picked boxes(N,4) : in corner-form and in pixels, where the presence of the class is over the threshold
    # labels(N)         : class id of the object in each picked box
    # probs(N)          : probability for the labelled object is present in the picked box
    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        images = images.to(self.device)
        with torch.no_grad():
            # self.timer.start()
            # boxes : shape (3000,4) is_test - corner-form box relative to image size;
            #                        not is_test (training) - center-form relative to priors.
            # scores: shape (3000,2) probability that each of the 3000 detection boxes contain each of the 2 classes
            # scores, boxes = self.net.forward(images)
            scores, boxes = self.net.forward(images)
            # print("Inference time: ", self.timer.end())
        # log.once('boxes {}-shape {}\n   scores {}-shape {}'.format(boxes,boxes.shape,scores,scores.shape))
        # boxes tensor([[[-0.0366, -0.0361,  0.1178,  0.1417],
        #  [-0.0834, -0.0815,  0.1257,  0.1974],
        #  [-0.0728, -0.0365,  0.1387,  0.1189],
        #  ...,
        #  [ 0.0357,  0.0659,  0.9744,  0.9441],
        #  [ 0.0503,  0.0770,  0.9612,  0.9314],
        #  [ 0.0336,  0.0293,  0.9741,  0.9556]]]) -shape torch.Size([1, 3000, 4])
        # scores tensor([[[0.9901, 0.0099],
        #  [0.9969, 0.0031],
        #  [0.9521, 0.0479],
        #  ...,
        #  [0.6604, 0.3395],
        #  [0.6656, 0.3344],
        #  [0.7257, 0.2743]]])-shape torch.Size([1, 3000, 2]) <= 2 candidate classes : background 0, my class 1.
        boxes = boxes[0]    # shape (3000,4)
        scores = scores[0]  # shape (3000,2)
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        # this version of nms is slower on GPU, so we move data to CPU.
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)): # skip background (class 0)
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            # box_probs (N, 5): boxes in corner-form and probabilities
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1) # e.g (10 boxes,4)+(10 boxes,1), dim=1 => (10,5)
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0)) # [1]*10 => [1,1,1,1,1,1,1,1,1,1]
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs) # cat([box_probs[10,5],box_probs[10,5]]) => box_probs[20,5]
        # picked_box_probs[0:4] : four corners; picked_box_probs[4] : probability
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]