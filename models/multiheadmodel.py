import torch.nn as nn
import torch

class MultiHeadModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict()
        self.num_features = backbone.encoder.fc.in_features

    def add_head(self, number: int, num_classes: int=2):
        self.heads[str(number)] = nn.Linear(self.num_features, num_classes)

    def expand_head(self, number: int=0, num_class_increment: int=2):
        key = str(number)
        old_head = self.heads[key]
        old_num_classes = old_head.out_features
        new_num_classes = old_num_classes + num_class_increment

        device = old_head.weight.device
        new_head = nn.Linear(self.num_features, new_num_classes).to(device)

        with torch.no_grad():
            new_head.weight[:old_num_classes] = old_head.weight
            new_head.bias[:old_num_classes]   = old_head.bias

        self.heads[key] = new_head

    def forward(self, x, number: int):
        features = self.backbone(x)
        return self.heads[str(number)](features)

    def save(self, path):
        torch.save(self.state_dict(), path)