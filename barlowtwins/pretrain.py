import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import ImageCollateFunction
from lightly.loss import BarlowTwinsLoss
from barlow_twins import BarlowTwins

from dataloader import *


from classifier import Classifier

# file paths
save_path = './'
file_name = "checkpoint.pth"

# Model
project_dimension = 2048
resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = BarlowTwins(backbone, project_dimension)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Data 
# data_location = '/tmp/unlabeled'
input_dimension = 32
# train_transform = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])

#data = LabeledDataset(root=data_location,split=split,transforms=train_transform)
#dataset = LightlyDataset.from_torch_dataset(data, transform=train_transform)
# dataset = LightlyDataset(
#     input_dir=data_location
# )

# collate_fn = ImageCollateFunction(input_size=input_dimension)
# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=2048,
#     collate_fn=collate_fn,
#     shuffle=True,
#     drop_last=True,
#     num_workers=8,
# )
train_dataset = UnlabeledDataset(root='/unlabeled', transform=torchvision.transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=input_dimension, shuffle=False, num_workers=8)


criterion = BarlowTwinsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)

print("Starting Training")
for epoch in range(2):
    total_loss = 0
    for (x0, x1), _, _ in dataloader:
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)
        loss = criterion(z0, z1)
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

torch.save(model.state_dict(), f"{save_path}/{file_name}")
