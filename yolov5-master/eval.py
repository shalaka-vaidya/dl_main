import torch
from dataset import LabeledDataset, UnlabeledDataset
import transforms as T
import utils

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

valid_dataset = LabeledDataset(root='/labeled', split="validation", transforms=get_transform(train=False))
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)
count=0
for img,trg in valid_loader:
    count+=1
    if count==3:
        break
    print(img[0].shape)
    print(trg[0])