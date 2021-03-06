import torch
import os
import math

from dataset import get_loader
from models import DeepPrior, EmaModel
from trainer import train

import argparse

def get_args(args=None):
    parser = argparse.ArgumentParser(description='argparse')
    model_group = parser.add_argument_group(description='deep prior')
    model_group.add_argument("--n_order", type=int, default=2,
                             help="order for reference prior")
    model_group.add_argument("--K", type=int, default=4,
                             help="number of particles")
    model_group.add_argument("--alpha", type=float, default=0.05,
                             help="weight of pred_entropy")
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
    model_group.add_argument("--use_mixup", type=int, default=0,
                             help="if use mixup trick")
    model_group.add_argument("--mixup_alpha", type=float, default=0.75,
                             help="mixup alpha")
    model_group.add_argument("--no_gradient_stop", type=int, default=0,
                             help="if stop gradient from weak augment")
    model_group.add_argument("--label_thresh", type=float, default=0.95,
                             help="label threshold for weak augment")
    model_group.add_argument("--no_jensen", type=int, default=0,
                             help="if use jensen inequality in computing H_yx")
    model_group.add_argument("--num_ops", type=str, default="2,2",
                             help="list of number of ops in strong augmentation")
    model_group.add_argument("--magnitude", type=str, default="10,10",
                             help="list of magnitude in strong augmentation")
    model_group.add_argument("--e_step", type=int, default=1024,
                             help="number of steps per epoch")
    model_group.add_argument("--epochs", type=int, default=50,
                             help="number of epochs")
    model_group.add_argument("--e_warmup", type=int, default=0,
                             help="number of warmup epochs")
    model_group.add_argument("--main_device", type=int, default=3,
                             help="main cuda device")
    model_group.add_argument("--skip_devices", type=str, default="-1",
                             help="list of unavailable cuda devices")
    model_group.add_argument("--eval_particle", type=bool, default=True,
                             help="if evaluate each particle")
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
    args.skip_devices = [int(d) for d in args.skip_devices.split(",")]
    args.num_ops = [int(d) for d in args.num_ops.split(",")]
    args.magnitude = [int(d) for d in args.magnitude.split(",")]

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
    print(f"ema_decay {args.ema_decay}")
    print(f"bn_momentum {args.bn_momentum}")
    print(f"train_prior {args.train_prior}")
    print(f"Hy_use_weak {args.Hy_use_weak}")
    print(f"use_mixup {args.use_mixup}")
    print(f"mixup_alpha {args.mixup_alpha}")
    print(f"no_gradient_stop {args.no_gradient_stop}")
    print(f"label_thresh {args.label_thresh}")
    print(f"no_jensen {args.no_jensen}")
    print(f"num_ops {args.num_ops}")
    print(f"magnitude {args.magnitude}")
    print(f"eval_particle {args.eval_particle}")
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
    def create_warmup_scheduler(optimizer, num_cycles=7./16., last_epoch=-1):
        num_warmup_steps = args.e_warmup * args.e_step
        num_training_steps = args.epochs * args.e_step
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))
    
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)

    scheduler = create_warmup_scheduler(optimizer)
    scaler = torch.cuda.amp.GradScaler()
    
    model, metrics = train(model, optimizer, scheduler, scaler, args,
                           train_loader_sp, train_loader_unsp, val_loader)