import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SwAV
#from pl_bolts.models.self_supervised.swav.transforms import (
#     SwAVTrainDataTransform, SwAVEvalDataTransform
# )
#from pl_bolts.transforms.dataset_normalizations import stl10_normalization

# data
batch_size = 128
train_dataset = UnlabeledDataset(root='/unlabeled', transform=torchvision.transforms.ToTensor())
dm = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#dm = STL10DataModule(data_dir='.', batch_size=batch_size)
# dm.train_dataloader = dm.train_dataloader_mixed
# dm.val_dataloader = dm.val_dataloader_mixed

# dm.train_transforms = SwAVTrainDataTransform(
#     normalize=stl10_normalization()
# )

# dm.val_transforms = SwAVEvalDataTransform(
#     normalize=stl10_normalization()
# )

# model
model = SwAV(
    gpus=1,
    num_samples=100,
    batch_size=batch_size
)

# fit
trainer = pl.Trainer(precision=16, max_epochs=5)
trainer.fit(model)