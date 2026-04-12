#For Seq-CIFAR-10 and Tiny-ImageNet datasets, we use
#ResNet-18 (not pretrained) as a base encoder for representation
# learning followed by a 2-layer projection MLP which
#maps representations to a 128-dimensional latent space [44].
#The hidden layer of projection MLP consists of 512 hidden
#units.
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class BackBone(nn.Module):
    def __init__(self, proj_hidden=512, proj_out=128):
        super().__init__()

        self.proj_hidden = proj_hidden
        
        self.encoder = models.resnet18(weights=None) # No pretrained
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