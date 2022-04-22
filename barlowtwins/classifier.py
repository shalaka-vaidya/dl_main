import torch
from torch import nn
from torchsummary import summary

class Classifier(nn.Module):

    def __init__(self, num_classes, embedding_network,weight_path,embedding_out_dim=2048):
        super().__init__()
        self.embedding_network = embedding_network
        self.embedding_network.load_state_dict(torch.load(weight_path))
        self.linear_classifier = nn.ModuleList([nn.Linear(embedding_out_dim, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, num_classes)])
#        summary(self.embedding_network, (3, 32, 32))


    def forward(self, x):
        x = self.embedding_network(x)
        for layer in self.linear_classifier:
            x = layer(x)

        return x