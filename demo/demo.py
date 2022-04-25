# Feel free to modifiy this file. 
# It will only be used to verify the settings are correct 
# modified from https://pytorch.org/docs/stable/data.html

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from barlow_twins import BarlowTwins
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


import transforms as T
import utils
from engine import train_one_epoch, evaluate


from dataset import UnlabeledDataset, LabeledDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes):
    resnet = torchvision.models.resnet50()
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = BarlowTwins(backbone, 2048)
    checkpoint_wt = torch.load("./latest.pth")
    backbone.load_state_dict(checkpoint_wt)
    modules = list(backbone.children())[:-1]
    backbone = torch.nn.Sequential(*modules)

    backbone.out_channels = 2048

    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,weights_backbone = checkpoint_wt)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("YAY")
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_path = './'
    file_name1 = "checkpoint_class_"
    num_classes = 101
    train_dataset = LabeledDataset(root='/labeled', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        torch.save(model.state_dict(), f"{save_path}/{file_name1+str(epoch)+'.pth'}")
        evaluate(model, valid_loader, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
