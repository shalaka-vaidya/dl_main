import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
import torchvision
import torch
from dataset import UnlabeledDataset, LabeledDataset
from pl_bolts.models.self_supervised.swav.transforms import (
    SwAVTrainDataTransform, SwAVEvalDataTransform
)
#from pl_bolts.transforms.dataset_normalizations import stl10_normalization

# data
batch_size = 128
train_dataset = UnlabeledDataset(root='/unlabeled', transform=torchvision.transforms.ToTensor())
dm = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#dm = STL10DataModule(data_dir='.', batch_size=batch_size)
# dm.train_dataloader = dm.train_dataloader_mixed
# dm.val_dataloader = dm.val_dataloader_mixed


dm.train_transforms = SwAVTrainDataTransform(
    normalize=torchvision.transforms.Normalize((0.4917,0.4694,0.4148),(0.2278,0.2240,0.2280))
)

dm.val_transforms = SwAVEvalDataTransform(
    normalize=torchvision.transforms.Normalize((0.4917,0.4694,0.4148),(0.2278,0.2240,0.2280))
)


# model
model = SwAV(
    gpus=1,
    num_samples=100,
    batch_size=batch_size
)

# fit
trainer = pl.Trainer(default_root_dir='./model',precision=16, max_epochs=1)
trainer.fit(model, dm)