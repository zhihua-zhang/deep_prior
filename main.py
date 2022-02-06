
import os
try:
    os.chdir("./workspace/research/repo")
except:
    pass

import torch

from dataset import get_loader
from models import DeepPrior, ModelEma
from trainer import train
from utils import report

import dataset, models, utils, trainer
import importlib
importlib.reload(models)


device = "cuda:0" if torch.cuda.is_available() else "cpu"

###############################################################################
# (a) Load data & Preprocess
###############################################################################

train_loader_unsp, train_loader_sp, val_loader = get_loader(order=2, split=1)


###############################################################################
# (b) Build Deep neural models
###############################################################################

K = 4
n_order = 2
alpha = 1.0
in_dim = 784
out_dim = 10
hidden_dims = [128, 32]

model = DeepPrior(in_dim, out_dim, hidden_dims, n_order, K, alpha).to(device)
model_ema = ModelEma(model)
model_ema.set(model)

###############################################################################
# (c) Train reference prior
###############################################################################

if __name__ == "__main__":
    base_lr = 1e-3
    weight_decay = 1e-5
    epochs = 20

    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: 0.95)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model, metrics = train(model, model_ema, optimizer, epochs, train_loader_unsp, train_loader_sp, val_loader)

    # report(metrics, epochs)