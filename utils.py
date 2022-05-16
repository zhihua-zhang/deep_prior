import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
import torchvision
import torchvision.transforms as T

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor):
        return x if random.random() > self.p else self.fn(x)

def transform_train_sp(args):
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

def transform_train_unsp_weak(args):
    return T.Compose([
        T.RandomHorizontalFlip(), #p=0.5
        T.RandomCrop(size=32,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

def transform_train_unsp_strong(args):
    return T.Compose([
        T.RandomHorizontalFlip(), #p=0.5
        T.RandomCrop(size=32,
                    padding=int(32*0.125),
                    padding_mode='reflect'),
        T.RandAugment(num_ops=2, magnitude=10),
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    return T.RandomApply(
        nn.ModuleList([
            T.RandomResizedCrop(image_size, scale=(0.7,1.0)),
            T.RandomAffine(30),
            T.RandomPerspective(p=1.0),
            T.GaussianBlur((3,3), sigma=(0.1,1.0)),
        ]), p=0.5
    )

def transform_val(args):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

def mixup_batch(inp, target, args):
    beta_sample = np.random.beta(args.mixup_alpha, args.mixup_alpha)

    # Trick to bias towards sample 1
    beta_sample = max(beta_sample, 1 - beta_sample)
    index = torch.randperm(inp.shape[0])

    mixed_inp = beta_sample * inp + (1 - beta_sample) * inp[index]
    mixed_target = beta_sample * target + (1 - beta_sample) * target[index]
    return mixed_inp, mixed_target
    
def viz(imgs, dirs="test", name="test.jpg"):
    import os
    from torchvision.utils import save_image, make_grid
    os.makedirs(dirs, exist_ok=True)
    nrow = int(len(imgs) ** 0.5)
    save_image(make_grid(imgs.float(), nrow=nrow, normalize=True), f"{dirs}/{name}")

def report(metrics, epochs, n_report=200):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    e_steps = np.linspace(0, epochs, n_report)
    
    steps = np.linspace(0, len(metrics['train']['objective'])-1, n_report).astype(int)
    ax[0].plot(e_steps, np.array(metrics['train']['objective'])[steps], label="objective")
    ax[0].plot(e_steps, np.array(metrics['train']['MI'])[steps], label="MI")
    ax[0].plot(e_steps, np.array(metrics['train']['logLikelihood'])[steps], label="logLikelihood")
    ax[0].plot(e_steps, np.array(metrics['train']['sp_logLikelihood'])[steps], label="sp_logLikelihood")
    ax[0].set_xlabel("Steps")
    ax[0].set_title("Train results")
    ax[0].legend()
    
    steps = np.linspace(0, len(metrics['train']['acc'])-1, n_report).astype(int)
    ax[1].plot(e_steps, np.array(metrics["train"]["acc"])[steps], label="train")
    ax[1].plot(e_steps, np.array(metrics["val"]["acc"])[steps], label="val")
    ax[1].plot(e_steps, np.array(metrics["train"]["sp_acc"])[steps], label="sp_train")
    ax[1].plot(e_steps, np.array(metrics["val"]["sp_acc"])[steps], label="sp_val")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy evaluation")
    ax[1].legend()
    
    fig.savefig("./res.png")
    plt.close('all')

def save_results(model, metrics, epoch, args, prefix=""):
    suffix = os.path.join(prefix, f"n_sp={args.n_sp}", f"alpha={args.alpha},gamma={args.gamma},K={args.K},order={args.n_order}")
    checkpoints_dir = os.path.join("checkpoints", suffix)
    if os.path.exists(checkpoints_dir):
        os.system(f"rm -rf {checkpoints_dir}")
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, "epoch={}-acc={:.3f}.pth".format(epoch, metrics["val"]["acc"][-1])))

    logs_dir = os.path.join("logs", suffix)
    if os.path.exists(logs_dir):
        os.system(f"rm -rf {logs_dir}")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, "epoch={}.json".format(epoch)), "w") as f:
        json.dump(metrics, f)