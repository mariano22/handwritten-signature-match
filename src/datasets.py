import glob
import re
import os
import torchvision
import torch

# Sigcomp 2009

SIGCOMP2009_FILE_PATHS = ['./data/sigcomp2009/NISDCC', './data/sigcomp2009/NFI']

SIGCOMP2009_REGEXES = [ r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.', r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).' ]

SIGCOMP2009_REGEX_Y_GROUP = [ 2, 3 ]

SIGCOMP2009_CLASS_TAGS = [ 'NISDCC-', 'NFI-' ]
    
def sigcomp2009(only_orignals=True, train=True):
    def match(fp, i): 
        return re.match(SIGCOMP2009_REGEXES[train], os.path.basename(fp)).group(i)
            
    files = [f for f in glob.glob(SIGCOMP2009_FILE_PATHS[train]+'/*') if f.endswith('png') or f.endswith('PNG') ]
    if only_orignals:
        files = [ f for f in files if match(f, SIGCOMP2009_REGEX_Y_GROUP[train]) ==  match(f, 1)]
    cls = [ SIGCOMP2009_CLASS_TAGS[train] + match(f, SIGCOMP2009_REGEX_Y_GROUP[train]) for f in files ]
    return list(zip(files,cls))

# MNIST

def mnist():
    ds = torch.utils.data.ConcatDataset([
        torchvision.datasets.MNIST('./data/mnist/', download=True),
        torchvision.datasets.MNIST('./data/mnist/', download=True, train=False),
    ])
    
    return ds

# Omniglot

OMNIGLOT_PATHS = [ './data/omniglot/images_background', './data/omniglot/images_evaluation' ]

def omniglot(valid=False):
    files = [f for f in glob.glob(OMNIGLOT_PATHS[valid]+'/**/**/*') if f.endswith('png') or f.endswith('PNG') ]
    cls = [ '-'.join(f.split('/')[-3:-1]) for f in files ]
    return list(zip(files,cls))

# CEDAR

CEDAR_FILE_PATH = './data/cedar'

def cedar():
    files = [f for f in glob.glob(CEDAR_FILE_PATH+'/*') if f.endswith('png') or f.endswith('PNG') ]
    cls = [ 'CEDAR-'+f.split('_')[1] for f in files ]
    return list(zip(files,cls))