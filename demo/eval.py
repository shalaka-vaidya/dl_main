import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
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

def get_model(num_classes,device):
    # torch.load(checkpoint_location)
    # resnet = torchvision.models.resnet50()
    # backbone = nn.Sequential(*list(resnet.children())[:-1])
    # backbone = BarlowTwins(backbone, 2048)
    #checkpoint_wt = torch.load('/home/azureuser/dl-project/dl_main/demo/checkpoint_25.pth',map_location=torch.device(device))
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)
    #model_child_list = [c for c  in model.backbone.children()]
    #model.backbone.body.load_state_dict(state_dict=checkpoint_wt, strict = False)
    #print(model.backbone)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("YAY")
    return model
    #return model

def get_model_new(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def main(checkpoint_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 101
    valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    #model = get_model(num_classes)
    model = get_model_new(num_classes)
    model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    args=str(sys.argv)
    checkpoint_path = args[1]
    main(checkpoint_path)
