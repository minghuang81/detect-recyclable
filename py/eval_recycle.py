import torch
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer
import argparse
import pathlib
import numpy as np
import log
import sys

# python eval_weed.py --trained_model models/trainingOutputModel.pth  --label_file models/trainingOutputModel-labels.txt --dataset ../datasets/open_images/Broccoli --eval_out /tmp
# output of the evaluation results:
# - <eval_out> / det_test_{class_name}.txt: contains detected boxes and in which image, one line per detected box
#                                           image-id probability xmin ymin xmax ymax

parser = argparse.ArgumentParser(description="SSD Evaluation on images in 'test' subdirectory of open_images Dataset.")
parser.add_argument("--trained_model", type=str)

parser.add_argument("--dataset", type=str, help="The root directory of the Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=False)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_out", default="/tmp", type=str, help="The directory to store evaluation results.")

args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


# return:
# true_case_stat : dict of per-class counters of not-difficult boxes (boxed objects) in that class
# all_gt_boxes   : all ground truth boxes arranged according to the hierachy of
#                       class index i  <- image id i1 <= [gt_box11, gt_box12, ...] 
#                                      <- image id i2 <= [gt_box21, gt_box22, ...]
#                                      ...
# all_difficult_cases : difficult or not-difficult attribute of a box (or boxed object)
#                       class index i  <- image id i1 <= [difficulty of box11, difficulty of box12, ...] 
#                                      <- image id i2 <= [difficulty of box21, difficulty of box22, ...]
#                                      ...
# all_gt_boxes and all_difficult_cases have the same number of entries : one value (difficulty) against one box.
def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            #ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,ClassId,ClassName
            t = line.rstrip().split(",")
            image_ids.append(t[0])
            scores.append(float(t[3]))          
            box = torch.tensor([float(t[4]),float(t[6]),float(t[5]),float(t[7])]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]            
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_out)
    if (not eval_path.is_dir):
        log.info('Trying to create {}'.format(eval_path))
        eval_path.mkdir(parents=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    dataset = OpenImagesDataset(args.dataset, dataset_type="test")  # test/xxxx.jpg

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    # log.info('true_case_stat {}, all_gb_boxes {}, all_difficult_cases {}, '.format(true_case_stat, all_gb_boxes, all_difficult_cases ))
    # true_case_stat {1: 182}
    # all_gb_boxes {1: {'0334271ebc0263d6': tensor([[  0.0000, 152.4311, 338.1105, 400.7900]]), 
    #                   '034a2ea7a2cc265f': tensor([[ 507.8589,  201.3688, 1023.0487,  562.2672]]),...
    #                   'fb60ad1a17610610': tensor([[   1.9108,    0.0000, 1022.0892, 1024.0000],
    #                                               [   8.9108,    0.0000, 1025.0892,   24.0000]])}}
    # all_difficult_cases {1: {'0334271ebc0263d6': [0], '034a2ea7a2cc265f': [0], ...'fb60ad1a17610610': [0, 0]}}                  

    net = create_mobilenetv1_ssd(len(class_names), is_test=True)

    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print('It took {} seconds to load the model.'.format(timer.end("Load Model")))
    predictor = create_mobilenetv1_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)

    results = []
    for i in range(len(dataset)):
        # print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        # print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.cumul("Load Image")
        timer.start("Predict")
        # top_k=1 only keep one detection per class with highest confidence 
        # prob_threshold discard the detections with lower confidence.
        # boxes(N,4), labels(N), probs(N) - N typically ~ 50 per image
        boxes, labels, probs = predictor.predict(image,top_k=1, prob_threshold=0.5)
        if (labels.size(0) == 0):   # no detection
            continue
        # print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        timer.cumul("Predict")
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            # boxes + 1.0  # matlab's indexes start from 1
            boxes.reshape(-1,4) + 1.0  # matlab's indexes start from 1
        ], dim=1))
    # len(results) = len(dataset) number of images in the Test set; results[-].shape : (N,7)
    # 7 = 1 image index + 1 class id + 1 prob (confidence to have the class) + 4 box corners
    # after torch.cat, results.shape : (N*len(dataset), 7)
    print("len(results)= {}".format(len(results)))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = "{}/det_test_{}.txt".format(eval_path,class_name)
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                # prob_box = sub[i, 2:].numpy()
                # image_id = dataset.ids[int(sub[i, 0])]
                # print>>f, image_id + " " + " ".join([str(v) for v in prob_box])
                #ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,ClassId,ClassName
                # ImageID = dataset.ids[int(sub[i, 0])]
                ImageID = dataset.data[int(sub[i, 0])]['image_id']
                LabelName = class_name
                Confidence = sub[i, 2].numpy()
                XMin = sub[i, 3].numpy()
                YMin = sub[i, 4].numpy()
                XMax = sub[i, 5].numpy()
                YMax = sub[i, 6].numpy()
                print>>f, '{},xxx,{},{},{},{},{},{},0,0,0,0,0,{},{}'.format(ImageID,LabelName,Confidence,XMin,XMax,YMin,YMax,LabelName,LabelName)

    print("results are printed into file \n\t{} \nformat:".format(prediction_path))
    print("\tImageID,xxx,LabelName,Confidence,XMin,XMax,YMin,YMax,0,0,0,0,0,LabelName,LabelName")

    aps = []
    print("Average Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = "{}/det_test_{}.txt".format(eval_path,class_name)
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print("\t{}: {}".format(class_name,ap))

    print("Average Precision Across All Classes:\n\t{}".format(sum(aps)/len(aps)))
    print("Average Time for loading an image:\n\t{}".format(timer.getAvg('Load Image')))
    print("Average Time for prediction:\n\t{}".format(timer.getAvg('Predict')))
    




