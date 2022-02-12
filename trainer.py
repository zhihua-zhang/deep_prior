import torch
from torch import nn
from utils import save_results

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate(model, train_loader_sp, val_loader=None):
    model.eval()
    loader = train_loader_sp if not val_loader else val_loader
    
    n_total = 0
    correct = 0
    with torch.no_grad():
        for data_sp, targets in loader:
            data_sp, targets = map(lambda x: x.to(device), [data_sp, targets])
            out_sp = model(data_sp)
            prob = out_sp[range(len(out_sp)), targets.flatten()].mean(dim=-1)
            pred = torch.where(prob > 0.5, torch.ones(1, device=device), torch.zeros(1, device=device))
            correct += (pred.flatten() == targets.flatten()).sum().item()
            n_total += len(targets)
    acc = correct / n_total
    
    return acc

def train(model, model_ema, optimizer, args, train_loader_unsp, train_loader_sp, val_loader):
    
    epochs = args.epochs
    e_step = args.e_step
    bz_actual = args.bz_actual
    gamma = args.gamma
    
    step = 0
    total_steps = epochs * e_step
    report_freq = 10
    val_freq = 500
    
    metrics = {
        "train": {
            "objective": [],
            "MI": [],
            "logLikelihood": [],
            "acc": [],
        },
        "val": {
            "acc": [],
            "ema_acc": [],
        }
    }

    e = 0
    n_unsp = 0
    n_sp = 0
    MI = torch.zeros(1, device=device)
    logLikelihood = torch.zeros(1, device=device)
    while e < epochs:
        model.train()
        
        # read data
        data_unsp, _ = next(iter(train_loader_unsp))
        data_sp, targets = next(iter(train_loader_sp))

        # apply data augmentation
        with torch.no_grad():
            data_unsp, data_sp = map(model.augment, [data_unsp, data_sp])

        # move to device
        data_unsp, data_sp, targets = map(lambda x: x.to(device), [data_unsp, data_sp, targets])

        # calc objective
        out_unsp = model(data_unsp)
        out_sp = model(data_sp)
        MI_i = model.get_MI(out_unsp)
        logLikelihood_i = model.get_logLikelihood(out_sp, targets)

        # update statistics
        n_unsp += len(data_unsp)
        n_sp += len(data_sp)
        MI = MI + MI_i
        logLikelihood = logLikelihood + logLikelihood_i
        
        # weight update
        if n_unsp >= bz_actual:
            MI = MI / n_unsp
            logLikelihood = logLikelihood / n_sp
            objective = gamma * MI + logLikelihood
            
            optimizer.zero_grad()
            loss = -objective
            loss.backward()
            optimizer.step()
            
            metrics["train"]["MI"].append(MI.item())
            metrics["train"]["logLikelihood"].append(logLikelihood.item())
            metrics["train"]["objective"].append(objective.item())

            # verbose
            if (step+1) % report_freq == 0:
                print("Epoch {}, Step {}/{}, obj: {:.4f}, MI: {:.4f}, log_likelihood: {:.4f}".format(
                    e+1, step+1, total_steps, objective.item(), MI.item(), logLikelihood.item()))
            
            # evaluation
            if (step+1) % val_freq == 0:
                train_acc = evaluate(model, train_loader_sp, None)
                val_acc = evaluate(model, None, val_loader)
                val_ema_acc = evaluate(model_ema, None, val_loader)
                metrics["train"]["acc"].append(train_acc)
                metrics["val"]["acc"].append(val_acc)
                metrics["val"]["ema_acc"].append(val_ema_acc)
                
                # verbose
                print("Epoch {}, Step {}/{}, train acc: {:.4f}, val acc: {:.4f}, val ema acc: {:.4f}".format(
                    e+1, step+1, total_steps, train_acc, val_acc, val_ema_acc))

            # reset
            n_unsp = 0
            n_sp = 0
            MI = torch.zeros(1, device=device)
            logLikelihood = torch.zeros(1, device=device)
            
            if (step + 1) % e_step == 0:
                e += 1
                # EMA weight update
                model_ema.update(model)
            
                if e % args.save_freq == 0:
                    save_results(model, model_ema, metrics, e)
            
            step += 1
        
        # lr_scheduler.step()
    
    return model, model_ema, metrics