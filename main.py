import torch
import os
import math

from dataset import get_loader
from models import build_BaseModel, DeepPrior, EmaModel
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
    model_group.add_argument("--alpha", type=float, default=0.05,
                             help="weight of pred_entropy")
    model_group.add_argument("--gamma", type=float, default=0.5,
                             help="weight of mutual info in objective")
    model_group.add_argument("--tau", type=float, default=1/3,
                             help="weight of weak augmentation")
    model_group.add_argument("--n_sp", type=int, default=25,
                             help="number of supervised samples per class")
    model_group.add_argument("--base_lr", type=float, default=0.03,
                             help="base learning rate")
    model_group.add_argument("--weight_decay", type=float, default=1e-4,
                             help="l2 weight decay")
    model_group.add_argument("--ema_decay", type=float,
                             default=0.999, help="ema decay rate of model ensemble")
    model_group.add_argument("--bn_momentum", type=float,
                             default=0.001, help="batch norm momentum")
    model_group.add_argument("--bz_sp", type=int, default=64,
                             help="labeled batch size in pytorch loader")
    model_group.add_argument("--train_prior", type=int, default=0,
                             help="if prior is trainable")
    model_group.add_argument("--Hy_use_weak", type=int, default=0,
                             help="use only weak augment when computing H_y")
    model_group.add_argument("--Hyw_use_order", type=int, default=0,
                             help="if use order when computing H_yx")
    model_group.add_argument("--Hyw_rescale", type=int, default=0,
                             help="if rescale H_yx by mask_ratio")
    model_group.add_argument("--use_mixup", type=int, default=0,
                             help="if use mixup trick")
    model_group.add_argument("--mixup_alpha", type=float, default=0.75,
                             help="mixup alpha")
    model_group.add_argument("--e_step", type=int, default=1024,
                             help="number of steps per epoch")
    model_group.add_argument("--epochs", type=int, default=100,
                             help="number of epochs")
    model_group.add_argument("--run_epochs", type=int, default=50,
                             help="number of epochs to actually run")
    model_group.add_argument("--e_warmup", type=int, default=0,
                             help="number of warmup epochs")
    model_group.add_argument("--main_device", type=int, default=3,
                             help="main cuda device")
    model_group.add_argument("--skip_devices", type=list, default=[],
                             help="unavailable cuda device list")
    model_group.add_argument("--save_prefix", type=str, default="",
                             help="path prefix when saving models")
    model_group.add_argument("--reload_dir", type=str, default="",
                             help="reload model dir")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args()
    args.gamma = 1 / (1 - args.tau**2)
    args.ema_decay_rate = 1 / args.epochs
    args.skip_devices = [int(d) for d in args.skip_devices]

    print(args.skip_devices)

    print("="*28)
    print(f"n_order: {args.n_order}")
    print(f"K: {args.K}")
    print(f"alpha: {args.alpha}")
    print(f"tau: {args.tau}")
    print(f"gamma: {args.gamma}")
    print(f"n_sp: {args.n_sp}")
    print(f"sp batch size: {args.bz_sp}")
    print(f"learning rate: {args.base_lr}")
    print(f"weight decay: {args.weight_decay}")
    print(f"#sp samples per class: {args.n_sp}")
    print(f"ema decay rate {args.ema_decay_rate}")
    print(f"bn_momentum {args.bn_momentum}")
    print(f"train_prior {args.train_prior}")
    print(f"Hy_use_weak {args.Hy_use_weak}")
    print(f"Hyw_use_order {args.Hyw_use_order}")
    print(f"Hyw_rescale {args.Hyw_rescale}")
    print(f"use_mixup {args.use_mixup}")
    print(f"mixup_alpha {args.mixup_alpha}")
    print("="*28)
    
    args.num_workers = 4
    args.tasks = list(range(10))
    args.seed = 42

    ###############################################################################
    # (a) Load data & Preprocess
    ###############################################################################
    train_loader_sp, train_loader_unsp, val_loader = get_loader(args)


    ###############################################################################
    # (b) Build Deep neural models
    ###############################################################################

    # num_classes = 2
    num_classes = 10
    n_blocks = [4, 4, 4]

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    dp_model = DeepPrior(num_classes, n_blocks, args)
    model = EmaModel(dp_model, decay=args.ema_decay)

    ###############################################################################
    # (c) Train reference priorG
    ###############################################################################
    
    lr = args.K * args.base_lr
    weight_decay = 5 / args.K * args.weight_decay
    optimizer = torch.optim.SGD(model.get_opt_params(wd=weight_decay), lr=lr, momentum=0.9, nesterov=True)
    # optimizer_sp = torch.optim.SGD(model_sp.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    def create_warmup_scheduler(optimizer, num_cycles=7./16., last_epoch=-1):
        num_warmup_steps = args.e_warmup * args.e_step
        num_training_steps = args.epochs * args.e_step
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            # return max(0., math.cos(math.pi * num_cycles * no_progress))
            return 0.5 * (1 + math.cos(math.pi * no_progress))
    
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)

    scheduler = create_warmup_scheduler(optimizer)
    # scheduler_sp = create_warmup_scheduler(optimizer_sp)
    scaler = torch.cuda.amp.GradScaler()
    # scaler_sp = torch.cuda.amp.GradScaler()
    
    model, metrics = train(model, optimizer, scheduler, scaler, args,
                           train_loader_sp, train_loader_unsp, val_loader)