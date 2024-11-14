from torch.utils.data import Dataset
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tqdm

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

def show_pair(e):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2), facecolor='white')
    for ax,x in zip(axes,e[:2]):
        if isinstance(x,str):
            img = plt.cm.gray(mpimg.imread(x))
        else:
            img = x
        ax.imshow(img)
        ax.axis('off')  # Hide axes for a cleaner look
    plt.suptitle(['Mismatch','Match'][e[2]], fontsize=16)
    plt.tight_layout()
    plt.show() 


class PairDataset(Dataset):
    def __init__(self, ds, fix_pairing = False, seed = 22):
        self.rng = np.random.default_rng(seed)
        self.ds = ds
        self.gds = group_by_y(ds)
        ys = set(self.gds.keys())
        self.y_to_other_y = {y: list(ys.difference({y})) for y in ys}
        self.fix_pairing = []
        if fix_pairing:
            print("Fixing pairs (wait!)...")
            self.fix_pairing = [self.get(i) for i in tqdm.tqdm(range(len(self)))]
        
        
    def __len__(self):
        return len(self.ds)*2

    def __getitem__(self, idx):
        if self.fix_pairing:
            return self.fix_pairing[idx]
        return self.get(idx)

    def choose(self, xs):
        return xs[self.rng.choice(range(len(xs))).item()]
        
    def get(self, idx):
        x,y = self.ds[idx//2]
        is_match = bool(idx % 2)
        if is_match:
            y_other = y
        else:
            y_other = self.choose(self.y_to_other_y[y])
        x_other = self.choose(self.gds[y_other])
        return x,x_other,y==y_other
