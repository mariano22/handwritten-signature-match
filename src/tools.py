from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

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

def binarize_image(np_img):
    _, otsu_gauss= cv2.threshold(cv2.GaussianBlur(np_img,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return otsu_gauss

def aspect(x):
    a,b = PIL.Image.open(x).size
    return a/b
    
def ds_mean_aspect(ds):
    #return [aspect(x) for x,_ in ds]
    return np.mean([aspect(x) for x,_ in ds]).item()

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def get_y_vocab(ds):
    itos = list(set(y for _,y in ds))
    stoi = {y:i for i,y in enumerate(itos)}
    return itos, stoi

def shapes(l):
    return [x.shape for x in l]

def get_encoder_out_dim(encoder, sample):
    with torch.no_grad():
        return encoder(sample).shape[-1]
