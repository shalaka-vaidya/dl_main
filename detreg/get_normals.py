import os
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



from dataset import UnlabeledDataset, LabeledDataset

def main():
    train_dataset = UnlabeledDataset(root='/unlabeled', transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=8)

    r_means=[]
    b_means=[]
    g_means=[]
    r_stds=[]
    g_stds=[]
    b_stds=[]
    for i, output in enumerate(train_loader):
        image = output[0].type(torch.DoubleTensor) 
        r_means.append(image[0].mean())
        b_means.append(image[1].mean())
        g_means.append(image[2].mean())
        r_stds.append(image[0].std())
        b_stds.append(image[1].std())
        g_stds.append(image[2].std())
    r_means=torch.FloatTensor(r_means)
    b_means=torch.FloatTensor(b_means)
    g_means=torch.FloatTensor(g_means)
    r_stds=torch.FloatTensor(r_stds)
    b_stds=torch.FloatTensor(b_stds)
    g_stds=torch.FloatTensor(g_stds)
    print("1:", r_means.mean(), r_stds.mean())
    print("2:", b_means.mean(), b_stds.mean())
    print("3:", g_means.mean(), g_stds.mean())

    
 
    
if __name__ == "__main__":
    main()
