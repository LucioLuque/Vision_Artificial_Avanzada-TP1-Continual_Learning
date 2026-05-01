import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BackBone(nn.Module):
    def __init__(self, proj_hidden=512, proj_out=128):
        super().__init__()

        self.proj_hidden = proj_hidden
        
        self.encoder = models.resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
         
        self.projector = nn.Sequential( # Solo para preentrenamiento
            nn.Linear(512, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out)
        )

    def forward(self, x):
        return self.encoder(x)
    
    def get_projection(self, x):
        return F.normalize(self.projector(self.encoder(x)), dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)