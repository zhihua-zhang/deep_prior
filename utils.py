import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, Tensor
import torchvision.transforms as T

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor):
        return x if random.random() > self.p else self.fn(x)

def weak_augmentation(image_size=(224, 224)):
    return nn.Sequential(
    )
    return nn.Sequential(
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(padding=4),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )

def strong_transform(image_size=(224, 224)):
    return nn.Sequential(
        tf.Resize(size=image_size),
        RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
        aug.RandomGrayscale(p=0.2),
        aug.RandomHorizontalFlip(),
        RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
        aug.RandomResizedCrop(size=image_size),
        #RandomPatchDrop(size=image_size),
        #aug.RandomPerspective(),
        aug.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        ),
    )

def report(metrics, epochs, n_report=200):
    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    e_steps = np.linspace(0, epochs, n_report)
    
    steps = np.linspace(0, len(metrics['train']['objective'])-1, n_report).astype(int)
    ax[0].plot(e_steps, np.array(metrics['train']['objective'])[steps], label="objective")
    ax[0].plot(e_steps, np.array(metrics['train']['MI'])[steps], label="MI")
    ax[0].plot(e_steps, np.array(metrics['train']['logLikelihood'])[steps], label="logLikelihood")
    ax[0].set_xlabel("Steps")
    ax[0].set_title("Train results")
    ax[0].legend()
    
    steps = np.linspace(0, len(metrics['train']['acc'])-1, n_report).astype(int)
    ax[1].plot(e_steps, np.array(metrics["train"]["acc"])[steps], label="train")
    ax[1].plot(e_steps, np.array(metrics["val"]["acc"])[steps], label="val")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy evaluation")
    ax[1].legend()
    
    fig.savefig("./mnist_res.png")
    plt.close('all')

def save_results(model, model_ema, metrics, epoch):
    if os.path.exists("./checkpoints"):
        os.system("rm -rf ./checkpoints")
    os.makedirs("./checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "./checkpoints/epoch={}-acc={:.3f}.pth".format(epoch+1, metrics["val"]["acc"][-1]))
    torch.save(model_ema.state_dict(), "./checkpoints/ema-epoch={}-acc={:.3f}.pth".format(epoch+1, metrics["val"]["ema_acc"][-1]))

    if os.path.exists("./logs"):
        os.system("rm -rf ./logs")
    os.makedirs("./logs", exist_ok=True)
    with open("./logs/epoch={}.json".format(epoch+1), "w") as f:
        json.dump(metrics, f)