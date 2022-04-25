import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from barlow_twins import BarlowTwins
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


from dataset import UnlabeledDataset, LabeledDataset
import transforms as T
import utils
from engine import evaluate

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
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 101
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    model = get_model(num_classes)
    model.to(device)
    model.load_state_dict(torch.load('./chkpt.pth', map_location=torch.device('cpu')))
    model.eval()

    evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    checkpoint_path = "./chkpt.pth"

    main()
