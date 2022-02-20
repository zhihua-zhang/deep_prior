
import os
try:
    os.chdir("./workspace/research/repo")
except:
    pass

import torch

from dataset import get_loader
from models import build_BaseModel, DeepPrior, ModelEma
from trainer import train
from utils import report

import argparse

def get_args(args=None):
    parser = argparse.ArgumentParser(description='argparse')
    model_group = parser.add_argument_group(description='BYOL')
    model_group.add_argument("--n_order", type=int, default=2,
                             help="order for reference prior")
    model_group.add_argument("--K", type=int, default=4,
                             help="number of particles")
    model_group.add_argument("--e_step", type=int, default=1024,
                             help="number of steps per epoch")
    model_group.add_argument("--alpha", type=float, default=1.0,
                             help="weight of pred_entropy")
    model_group.add_argument("--gamma", type=float, default=1.0,
                             help="weight of mutual info in objective")
    model_group.add_argument("--bz_sp", type=int, default=64,
                             help="labeled batch size in pytorch loader")
    model_group.add_argument("--epochs", type=int, default=60,
                             help="number of epochs")
    model_group.add_argument("--base_lr", type=float, default=1e-3,
                             help="base learning rate")
    model_group.add_argument("--weight_decay", type=float, default=1e-5,
                             help="l2 weight decay")
    model_group.add_argument("--n_sp", type=int, default=500,
                             help="number of supervised samples per class")
    model_group.add_argument("--target_comb", type=str, default="cross_product",
                             help="combination of possible targets: 'cross_product' or 'permutations'")
    model_group.add_argument("--base_model", type=str, default="fc",
                             help="the type of base model: 'fc' or 'cnn'")
    model_group.add_argument("--lr_scheduler", type=list, default=[10, 20, 25],
                             help="lr scheduler")
    model_group.add_argument("--ema_decay_rate", type=float,
                             default=0.999, help="ema decay rate of model ensemble")
    model_group.add_argument("--save_freq", type=int, default=5,
                             help="frequency of model saving")
    model_group.add_argument("--model_pth", type=str, default="",
                             help="reload model path")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args()

    print("="*28)
    print(f"n_order: {args.n_order}")
    print(f"K: {args.K}")
    print(f"alpha: {args.alpha}")
    print(f"gamma: {args.gamma}")
    print(f"sp batch size: {args.bz_sp}")
    print(f"learning rate: {args.base_lr}")
    print(f"weight decay: {args.weight_decay}")
    print(f"#sp samples per class: {args.n_sp}")
    print(f"ema decay rate {args.ema_decay_rate}")
    print(f"target_comb {args.target_comb}")
    print(f"base_model {args.base_model}")
    print("="*28)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ###############################################################################
    # (a) Load data & Preprocess
    ###############################################################################
    args.base_model = 'cnn'
    train_loader_unsp, train_loader_sp, val_loader = get_loader(args)


    ###############################################################################
    # (b) Build Deep neural models
    ###############################################################################

    in_dim = 784
    out_dim = 10
    hidden_dims = [128, 32]

    # model = DeepPrior(in_dim, out_dim, hidden_dims, args).to(device)
    # model_ema = ModelEma(model, in_dim, out_dim, hidden_dims, args)

    model = DeepPrior(in_dim, out_dim, hidden_dims, args).to(device)
    model_sp = build_BaseModel(in_dim, out_dim, hidden_dims, args).to(device)
    model_ema = ModelEma(model, in_dim, out_dim, hidden_dims, args).to(device)

    ###############################################################################
    # (c) Train reference priorG
    ###############################################################################

    epochs = args.epochs

    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    optimizer_sp = torch.optim.SGD(model_sp.parameters(), lr=args.base_lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: 0.95)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model, model_ema, metrics = train(model, model_sp, model_ema, optimizer, optimizer_sp, args, train_loader_unsp, train_loader_sp, val_loader)


    # report(metrics, epochs)