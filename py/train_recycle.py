import argparse
import os
import log
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

# Continous training ...
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
# training data
parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
# Frequently modified training params
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--labelInModelOut_folder', default='./models/',
                    help='Directory for input classification labels and output checkpoint models')
args = parser.parse_args()

# hardcoded params for training
args.freeze_base_net = True
log.info("freeze_base_net : {}".format(args.freeze_base_net))
args.pretrained_ssd = "models/pretrained-ssdv1.pth"
log.info("Using pretrained model: {}".format(args.pretrained_ssd))
args.batch_size = 5
log.info("mini-batch size: {}".format(args.batch_size))
#args.num_workers = 4
args.num_workers = 1
log.info("num_workers used in dataloading: {}".format(args.num_workers))
args.validation_epochs = 5
log.info("the number epochs before each validation: {}".format(args.validation_epochs))
args.debug_steps = 100
log.info("debug log output frequency : one debug log per {} training samples".format(args.debug_steps))
args.labelInModelOut_folder = 'models/'
log.info("Output directory for classification labels and checkpoint models : {}".format(args.labelInModelOut_folder))
# hardcoded params for SGD
args.lr = 1e-2
log.info("initial learning rate : {}".format(args.lr))                  
args.extra_layers_lr = args.lr
log.info("initial learning rate for the layers not in base net and prediction heads. : {}".format(args.extra_layers_lr)) 
args.momentum = 0.9
log.info("Momentum value for optim : {}".format(args.momentum)) 
args.weight_decay = 5e-4
log.info("Weight decay for SGD : {}".format(args.weight_decay))
args.gamma = 0.1
log.info("Gamma update for SGD : {}".format(args.gamma))
args.scheduler = 'cosine'
log.info("Scheduler for SGD. It can one of multi-step and cosine : {}".format(args.scheduler))
args.t_max = 100
log.info("T_max value for Cosine Annealing Scheduler : {}".format(args.t_max))

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda:0")
    log.info("Use Cuda {}".format(DEVICE))
else:
    DEVICE = torch.device("cpu")
    log.info("Use CPU {}".format(DEVICE))

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        # log.once('images.shape={}, boxes.shape={}, labels.shape={}'.format( images.shape, boxes.shape, labels.shape))
        # - train_weed1.py:88 images.shape=torch.Size([5, 3, 300, 300]), boxes.shape=torch.Size([5, 3000, 4]), labels.shape=torch.Size([5, 3000])
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        # log.once('confidence.shape={}, locations.shape={}'.format( confidence.shape, locations.shape))
        # confidence.shape=torch.Size([5, 3000, 2]), locations.shape=torch.Size([5, 3000, 4])
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            log.info(
                "Epoch: {}, example: {}, "
                "Average all Loss: {}, "
                "Average Regression Loss {}, "
                "Average Classification Loss: {}"
                .format(epoch,i,avg_loss,avg_reg_loss,avg_clf_loss)
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    log.info(args)

    create_net = create_mobilenetv1_ssd
    config = mobilenetv1_ssd_config
    
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    log.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:

        dataset = OpenImagesDataset(dataset_path,
                transform=train_transform, target_transform=target_transform,
                dataset_type="train", balance_data=args.balance_data)
        label_file = os.path.join(args.labelInModelOut_folder, "outModel-labels.txt")
        store_labels(label_file, dataset.class_names)
        log.info(dataset)
        num_classes = len(dataset.class_names)

        datasets.append(dataset)
    log.info("Stored labels into file "+label_file)
    train_dataset = ConcatDataset(datasets)
    log.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    log.info("Prepare Validation datasets.")
    val_dataset = OpenImagesDataset(dataset_path,
                                    transform=test_transform, target_transform=target_transform,
                                    dataset_type="test")
    log.info(val_dataset)
    log.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    log.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    # freeze_base_net:
    log.info("Freeze base net..")
    freeze_net_layers(net.base_net)
    params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                net.regression_headers.parameters(), net.classification_headers.parameters())
    # log.info("params 1 = "+str(params))                         
    params = [
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': args.extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]
    # log.info("params 2 = "+str(params)) 

    timer.start("Load Model")
    if args.resume:
        log.info("Resume from the model "+args.resume)
        net.load(args.resume)
    else:
        log.info("Init from pretrained ssd "+args.pretrained_ssd)
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    log.info('Took '+str(timer.end("Load Model"))+' seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    log.info("Learning rate: "+str(args.lr) + ", Extra Layers learning rate: "+str(args.extra_layers_lr))

    log.info("Uses CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
 
    log.info("Start training from epoch "+str(last_epoch + 1))
    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step() # place scheduler.step() after training torch 1.1.0
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            log.info(
                "Epoch: {},"
                "Validation Loss: {}, "
                "Validation Regression Loss {}, "
                "Validation Classification Loss: {}"
                .format(epoch,val_loss,val_regression_loss,val_classification_loss)
            )
            model_path = os.path.join(args.labelInModelOut_folder, "outModel.pth")
            net.save(model_path)
            log.info("Saved model "+model_path)
