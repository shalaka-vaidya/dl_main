import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from barlow_twins import BarlowTwins
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import transforms as T
import utils
from engine import train_one_epoch, evaluate


from dataset import UnlabeledDataset, LabeledDataset

torch.manual_seed(3407)
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(num_classes,device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # Initializing backbone
    model_child_list = [c for c  in model.backbone.children()]
    checkpoint_wt= torch.load('../latest.pth')
    model_child_list[0].load_state_dict(state_dict=checkpoint_wt, strict = False)
    
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    
def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 101
    train_dataset = LabeledDataset(root='/labeled_data', split="training", transforms=get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=6, collate_fn=utils.collate_fn)

    valid_dataset = LabeledDataset(root='/labeled_data', split="validation", transforms=get_transform(train=False))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=5, shuffle=False, num_workers=6, collate_fn=utils.collate_fn)

    model = get_model(num_classes,device)
    model.to(device)

    #if resuming from checkpoint
    #model.load_state_dict(torch.load('checkpoint_class_new1.pth'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    save_path = './'
    file_name1 = "checkpoint_class_new"

    num_epochs = 20
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        print("Saving model")
        torch.save(model.state_dict(), f"{save_path}/{file_name1+str(epoch)+'.pth'}")
        evaluate(model, valid_loader, device=device)
    print("That's it!")

if __name__ == "__main__":
    main()
