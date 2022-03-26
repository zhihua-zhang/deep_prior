from re import A
import torch
from torch import nn
from utils import default_transform, save_results

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def evaluate_sp(model, loader):
    model.eval()
    with torch.no_grad():
        n_total = 0
        correct = 0
        for data_sp, targets in loader:
            data_sp, targets = map(lambda x: x.to(device), [data_sp, targets])
            out_sp = model(data_sp)
            pred = out_sp.argmax(dim=1)
            correct += (pred.flatten() == targets.flatten()).sum().item()
            n_total += len(targets)
        
        acc = correct / n_total
    return acc

def evaluate(model, train_loader_sp, val_loader):
    model.eval()
    with torch.no_grad():
        logLikelihood_sp = torch.zeros(len(model.deep_priors), device=device)
        sp_cnt = 0
        n_total = 0
        correct = 0
        for data_sp, targets in train_loader_sp:
            data_sp, targets = map(lambda x: x.to(device), [data_sp, targets])
            out_sp = model(data_sp)
            
            pred = out_sp.mean(dim=-1).argmax(dim=1)
            correct += (pred.flatten() == targets.flatten()).sum().item()
            n_total += len(targets)
            
            prob = out_sp[range(len(out_sp)), targets.flatten()]
            logLikelihood_sp = logLikelihood_sp + torch.log(prob).sum(dim=0)
            sp_cnt += len(data_sp)
        
        train_acc = correct / n_total
        
        logLikelihood_sp = logLikelihood_sp / sp_cnt
        Likelihood_sp = torch.exp(logLikelihood_sp)
        model.Likelihood_sp = Likelihood_sp.reshape(1,1,-1)
        
        n_total = 0
        correct = 0
        for data_sp, targets in val_loader:
            data_sp, targets = map(lambda x: x.to(device), [data_sp, targets])
            out_sp = model(data_sp)
            
            posterior_prob = out_sp * model.Likelihood_sp
            posterior_prob = posterior_prob / posterior_prob.sum(dim=1, keepdims=True)
            pred = posterior_prob.mean(dim=-1).argmax(dim=1)
            correct += (pred.flatten() == targets.flatten()).sum().item()
            n_total += len(targets)
        
        val_acc = correct / n_total
    return train_acc, val_acc

def train(model, model_sp, model_ema, optimizer, optimizer_sp, args, train_loader_unsp, train_loader_sp, val_loader):
    
    epochs = args.epochs
    e_step = args.e_step
    gamma = args.gamma
    augment = default_transform()
    
    step = 0
    total_steps = epochs * e_step
    report_freq = 100
    eval_freq = 1
    
    metrics = {
        "train": {
            "objective": [],
            "MI": [],
            "logLikelihood": [],
            "sp_logLikelihood": [],
            "acc": [],
            "sp_acc": []
        },
        "val": {
            "acc": [],
            "sp_acc": [],
            "ema_acc": [],
        }
    }

    MI = torch.zeros(1, device=device)
    logLikelihood = torch.zeros(1, device=device)
    for e in range(epochs):
        model.train()
        model_sp.train()

        # read data
        for _ in range(e_step):
            data_unsp, _ = next(iter(train_loader_unsp))
            data_sp, targets = next(iter(train_loader_sp))

            # apply data augmentation
            if args.use_augment:
                with torch.no_grad():
                    data_unsp, data_sp = map(augment, [data_unsp, data_sp])

            # move to device
            data_unsp, data_sp, targets = map(lambda x: x.to(device), [data_unsp, data_sp, targets])

            # calc objective
            out_unsp = model(data_unsp)
            out_sp = model(data_sp)
            
            MI = model.get_MI(out_unsp) / len(data_unsp)
            logLikelihood = model.get_logLikelihood(out_sp, targets) / len(data_sp)

            # weight update
            objective = gamma * MI + logLikelihood

            optimizer.zero_grad()
            loss = -objective
            loss.backward()
            optimizer.step()
            
            ### supervise only  ###
            out_sp_2 = model_sp(data_sp)
            prob2 = out_sp_2[range(len(out_sp_2)), targets.flatten()]
            logLikelihood2 = torch.mean(torch.log(prob2))
            optimizer_sp.zero_grad()
            loss2 = -logLikelihood2
            loss2.backward()
            optimizer_sp.step()
            ### --------------- ###
            
            metrics["train"]["MI"].append(MI.item())
            metrics["train"]["logLikelihood"].append(logLikelihood.item())
            metrics["train"]["objective"].append(objective.item())
            metrics["train"]["sp_logLikelihood"].append(logLikelihood2.item())

            # verbose
            if (step+1) % report_freq == 0:
                print("Epoch {}, Step {}/{}, obj: {:.4f}, MI: {:.4f}, log_likelihood: {:.4f}".format(
                    e+1, step+1, total_steps, objective.item(), MI.item(), logLikelihood.item()))
            
            step += 1
            
        # EMA weight update
        model_ema.update(model)
        
        # evaluation
        if (e+1) % eval_freq == 0:
            train_acc, val_acc = evaluate(model, train_loader_sp, val_loader)
            train_sp_acc = evaluate_sp(model_sp, train_loader_sp)
            val_sp_acc = evaluate_sp(model_sp, val_loader)
            _, val_ema_acc = evaluate(model_ema, train_loader_sp, val_loader)
            metrics["train"]["acc"].append(train_acc)
            metrics["val"]["acc"].append(val_acc)
            metrics["train"]["sp_acc"].append(train_sp_acc)
            metrics["val"]["sp_acc"].append(val_sp_acc)
            metrics["val"]["ema_acc"].append(val_ema_acc)
            
            # verbose
            print("Epoch {}, Step {}/{}, train acc: {:.4f}, val acc: {:.4f}, val ema acc: {:.4f}".format(
                e+1, step+1, total_steps, train_acc, val_acc, val_ema_acc))

            # verbose
            print("(sp only) Epoch {}, Step {}/{}, train acc: {:.4f}, val acc: {:.4f}".format(
                e+1, step+1, total_steps, train_sp_acc, val_sp_acc))
    
        if (e+1) % args.save_freq == 0:
            save_results(model, model_ema, metrics, e+1, args)
            
        # lr_scheduler.step()
    
    return model, model_ema, metrics