from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms as transforms

def group_by_y(ds):
    gds = defaultdict(list)
    for x,y in ds:
        gds[y].append(x)
    return gds

def split_ds(ds, is_valid):
    train,valid = [], []
    for x in ds:
        (train,valid)[is_valid(x)].append(x)
    return train,valid

# Make a tensor ready to be displayed
def showable(t):
    t = t.detach().cpu()
    # it assumes it was normalized with Imagenet stats
    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0,0,0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1,1,1]),
    ])
    if t.min()<0:
        t = inverse_transform(t)
    if t.size(0) == 3:
        t = t.permute(1, 2, 0)
    return t

def show_tensor(tensor):
    tensor = showable(tensor)
    plt.imshow(tensor)
    plt.axis('off')  # Hide axis
    plt.show()

def show_pair(e):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2), facecolor='white')
    for ax,x in zip(axes,e[:2]):
        if isinstance(x,str):
            img = plt.cm.gray(mpimg.imread(x))
        elif isinstance(x,torch.Tensor):
            img = showable(x)
        else:
            img = x
        ax.imshow(img)
        ax.axis('off')  # Hide axes for a cleaner look
    plt.suptitle(['Mismatch','Match'][e[2]], fontsize=16)
    plt.tight_layout()
    plt.show() 


