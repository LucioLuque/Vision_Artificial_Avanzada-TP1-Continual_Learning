import torch.nn as nn
import torch

class MultiHeadModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self.num_features = backbone.proj_out

    def add_head(self, number: int, num_classes: int=2):
        self.heads[str(number)] = nn.Linear(self.num_features, num_classes)

    def forward(self, x, number: int):
        features = self.backbone(x)
        return self.heads[str(number)](features)

    def save(self, path):
        torch.save(self.state_dict(), path)