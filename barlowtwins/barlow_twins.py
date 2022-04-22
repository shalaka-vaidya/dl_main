import torch
from torch import nn
from lightly.models.modules import BarlowTwinsProjectionHead

class BarlowTwins(nn.Module):
    def __init__(self, backbone, proj_dim=2048):
        super().__init__()
        self.backbone = backbone
        self.proj_dim = proj_dim
        self.projection_head = BarlowTwinsProjectionHead(2048, 2048, self.proj_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z





