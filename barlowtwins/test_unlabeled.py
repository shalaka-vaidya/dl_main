import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



from dataset import UnlabeledDataset, LabeledDataset


def main():
    train_dataset = UnlabeledDataset(root='/unlabeled', transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=8)

    for i, output in enumerate(train_loader):
        print(i, output.shape)

    
if __name__ == "__main__":
    main()
