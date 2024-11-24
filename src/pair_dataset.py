
from torch.utils.data import Dataset
import tqdm
import numpy as np

from src.tools import *

class PairDataset(Dataset):
    def __init__(self, ds, x_transform = None, y_transform = None,fix_pairing = False, seed = 22):
        self.rng = np.random.default_rng(seed)
        self.ds = ds
        self.gds = group_by_y(ds)
        ys = set(self.gds.keys())
        self.y_to_other_y = {y: list(ys.difference({y})) for y in ys}
        self.fix_pairing = []
        if fix_pairing:
            print("Fixing pairs (wait!)...")
            self.fix_pairing = [self.draw(i) for i in tqdm.tqdm(range(len(self)))]
        if x_transform:
            self.x_transform = x_transform
        else:
            self.x_transform = lambda x : x
        if y_transform:
            self.y_transform = y_transform
        else:
            self.y_transform = lambda x : x
        
        
    def __len__(self):
        return len(self.ds)*2

    def __getitem__(self, idx):
        x1,x2,y1,y2 = self.get_raw(idx)
        return self.x_transform(x1), self.x_transform(x2), self.y_transform(y1), self.y_transform(y2)
        
    def get_raw(self, idx):
        if self.fix_pairing:
            return self.fix_pairing[idx]
        return self.draw(idx)

    def choose(self, xs):
        return xs[self.rng.choice(range(len(xs))).item()]
        
    def draw(self, idx):
        x,y = self.ds[idx//2]
        is_match = bool(idx % 2)
        if is_match:
            y_other = y
        else:
            y_other = self.choose(self.y_to_other_y[y])
        x_other = self.choose(self.gds[y_other])
        return x,x_other,y,y_other