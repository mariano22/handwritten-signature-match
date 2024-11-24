import torchvision.transforms as transforms
from torchvision.transforms import v2
import torchvision
import torch
import PIL
import numpy as np

from src.tools import binarize_image, get_device, get_y_vocab

device = get_device()
lr = 5e-4
adam_eps = 1e-5

def x_transform_signatures():
    transform = transforms.Compose([
        torchvision.transforms.Lambda(PIL.Image.open),
        torchvision.transforms.Lambda(np.asarray),
        torchvision.transforms.Lambda(binarize_image), 
        torchvision.transforms.Lambda(torch.from_numpy),
        torchvision.transforms.Lambda(lambda t : t.unsqueeze(0)), # Add batch dimension
        torchvision.transforms.Resize((122,244)), # Resize all to same size for batching (with rectangular ratio 1x2)
        v2.RGB(), # Transform to RGB so we can use pretrained resnet
        v2.ToDtype(torch.float32, scale=True),
        torchvision.transforms.Lambda(lambda v : 1.0-v.float()), # Negate the image (black: background, white: signature)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize using Imagenet features (same as resnet pretrain)
    ])
    return transform

def y_transform_category(ds):
    itos,stoi = get_y_vocab(ds)
    return torchvision.transforms.Lambda(lambda ys : stoi[ys])

def x_transform_mnist():
    transform = transforms.Compose([
        torchvision.transforms.ToTensor(),
        v2.RGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        ])
    return transform

class SiamessClassifier(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        self.cls = torch.nn.Sequential(
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512,128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128,1),
        )
        
        self.encoder = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            torch.nn.Flatten(),
        )

    def forward(self, x1,x2, y1, y2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        logits = self.cls(f1-f2).squeeze(1)
        labels = (y1==y2).float()
        loss = self.criterion(logits, labels)
        return f1, f2, labels, logits, loss
        