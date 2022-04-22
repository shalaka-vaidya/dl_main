import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.loss import BarlowTwinsLoss
from barlow_twins import BarlowTwins
from torchvision.models.detection import FasterRCNN, _resnet_fpn_extractor

from dataloader import *

# file paths
save_path = './'
file_name = "checkpoint.pth"

# Model
project_dimension = 2048
resnet = torchvision.models.resnet50()
resnet.load_state_dict(torch.load(f"{save_path}/{file_name}"))
backbone = resnet
backbone = _resnet_fpn_extractor(backbone)
model = FasterRCNN(backbone, 100)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Data 
data_location = '/media/wlodder/Data/Datasets/Datasets/Image/neha/labeled_data/'
input_dimension = 32

split = "training"
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_dimension, input_dimension)),
    torchvision.transforms.ToTensor()
])

fine_tune_dataset = LabeledDataset(root=data_location,split=split,transforms=train_transform)
fine_tune_dataloader = torch.utils.data.DataLoader(
    fine_tune_dataset,
    batch_size=1,
    shuffle=True,
    drop_last=True,
    num_workers=8,
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

classifier = classifier.to(device)
classifier.eval()
classifier_loss = torch.nn.BCEWithLogitsLoss()
print("Starting Testing")
total_loss = 0
for x, y in fine_tune_dataloader:
    y = y['labels']
    print(y)
    y = y.to(device)
    y = torch.nn.functional.one_hot(y, 100)[0].float()
    x = x.to(device)
    y_pred = classifier(x)
    loss = classifier_loss(y, y_pred)
    total_loss += loss.detach()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
print(f"loss: {total_loss:.5f}")