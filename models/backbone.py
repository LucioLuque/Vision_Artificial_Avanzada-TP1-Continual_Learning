#For Seq-CIFAR-10 and Tiny-ImageNet datasets, we use
#ResNet-18 (not pretrained) as a base encoder for representation
# learning followed by a 2-layer projection MLP which
#maps representations to a 128-dimensional latent space [44].
#The hidden layer of projection MLP consists of 512 hidden
#units.
import torch
import torch.nn as nn
from torchvision import models

class BackBone(nn.Module):
    def __init__(self, proj_hidden=512, proj_out=128):
        super().__init__()

        self.proj_out = proj_out
        
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
         
        self.projector = nn.Sequential(
            nn.Linear(512, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out)
        )

    def forward(self, x):
        representation = self.encoder(x)
        projection = self.projector(representation)
        return projection

    def save(self, path):
        torch.save(self.state_dict(), path)