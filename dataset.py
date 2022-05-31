
import os
import re
import numpy as np
import random
from PIL import Image
import torch

from torchvision import datasets

from utils import transform_train_sp, transform_train_unsp_weak, transform_train_unsp_strong, transform_val

###############################################################################
# (a) Load data & Preprocess
###############################################################################


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, train=True,
                transform=None, target_transform=None,
                download=False):
        super().__init__(root, train=train,
                          transform=transform,
                          target_transform=target_transform,
                          download=download)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
  
class transform_unsp(object):
    def __init__(self, args, transform_unsp_weak, transform_unsp_strong):
        self.args = args
        self.weak = transform_unsp_weak
        self.strong = transform_unsp_strong

    def __call__(self, x):
        num_ops = random.randint(*self.args.num_ops)
        magnitude = random.randint(*self.args.magnitude)
        weak = self.weak(x)
        strong = self.strong(num_ops, magnitude)(x)
        return weak, strong

def get_loader(args, seed=2022):
    torch.manual_seed(seed)
    
    n_order = args.n_order
    bz_sp = args.bz_sp
    bz_unsp = 7 * bz_sp
    assert bz_unsp % n_order == 0, f"bz_unsp:{bz_unsp}, n_order:{n_order} - unsupervised batch size is not a multiple of n_order"
    
    train_data_sp = CIFAR10SSL(root="data/cifar10", train=True, transform=transform_train_sp())
    train_data_unsp = CIFAR10SSL(root="data/cifar10", train=True, transform=transform_unsp(args, transform_train_unsp_weak(), transform_train_unsp_strong))
    val_data = CIFAR10SSL(root="data/cifar10", train=False, transform=transform_val(), download=False)
    
    labels = np.array(train_data_sp.targets)
    rng = np.random.default_rng(args.seed)
    label_idx = []
    for t in args.tasks:
        loc = np.where(labels == t)[0]

        ### fix for comparison? ###
        loc = loc[:args.n_sp]
        # loc = rng.choice(loc, args.n_sp, False)
        label_idx.append(loc)
    label_idx = np.hstack(label_idx)
    assert len(label_idx) == (len(args.tasks) * args.n_sp)
    
    train_data_sp.data = train_data_sp.data[label_idx]
    train_data_sp.targets = np.array(train_data_sp.targets)[label_idx]
    # train_data_sp.targets = (np.array(train_data_sp.targets)[indexs] == 3).astype(int)
    
    train_loader_sp = torch.utils.data.DataLoader(train_data_sp, batch_size=bz_sp, shuffle=True, num_workers=args.num_workers)
    train_loader_unsp = torch.utils.data.DataLoader(train_data_unsp, batch_size=bz_unsp, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False, num_workers=args.num_workers)

    print("train dataset(unsupervise) n_sample:{n_sample}, shape:{shape}".format(n_sample=len(train_data_sp), shape=train_data_sp.data.shape))
    print("train dataset(supervise)   n_sample:{n_sample}, shape:{shape}".format(n_sample=len(train_data_unsp), shape=train_data_unsp.data.shape))
    print("val   dataset n_sample:{n_sample}, shape:{shape}".format(n_sample=len(val_data), shape=val_data.data.shape))
    print("train loader(unsupervise) bz:{bz}".format(bz=bz_unsp))
    print("train loader(supervise)   bz:{bz}".format(bz=bz_sp))
    print("val   loader bz:{bz}".format(bz=bz_sp))
    print("="*28)
    return train_loader_sp, train_loader_unsp, val_loader


# if __name__ == "__main__":
#     from main import get_args
#     args = get_args()
    
#     train_ds_unsp, train_ds_sp, val_ds = get_dataset(args)
#     train_loader_unsp, train_loader_sp, val_loader = get_loader(args)