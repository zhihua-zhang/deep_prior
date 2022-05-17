from re import A
import time, os, json
from collections import defaultdict
import torch
from torch import nn
from utils import mixup_batch, save_results

@torch.no_grad()
def evaluate_noprior(model, loader, use_ema=False):
    model.eval()
    device = model.main_device
    n_total = 0
    correct = 0
    for data, targets in loader:
        targets = targets.to(device, non_blocking=True)
        
        out = model(data, use_ema=use_ema)
        pred = out.mean(dim=-1).argmax(dim=1)
        correct += (pred.flatten() == targets.flatten()).sum().item()
        n_total += len(targets)
    
    acc = correct / n_total
    return acc

@torch.no_grad()
def evaluate(model, loader, use_ema=False):
    model.eval()
    device = model.main_device
    prior = model.get_prior(use_ema=use_ema).reshape(1,1,-1)
    n_total = 0
    correct = 0
    for data, targets in loader:
        targets = targets.to(device, non_blocking=True)
        
        out = model(data, use_ema=use_ema)
        pred = (out * prior).sum(dim=-1).argmax(dim=1)
        correct += (pred.flatten() == targets.flatten()).sum().item()
        n_total += len(targets)
    
    acc = correct / n_total
    return acc

@torch.no_grad()
def evaluate_bayes(model, train_loader_sp, val_loader, use_ema=False):
    model.eval()
    device = model.main_device
    
    logLikelihood_sp = torch.zeros(model.args.K, device=device)
    sp_cnt = 0
    n_total = 0
    correct = 0
    prior = model.get_prior(use_ema=use_ema).reshape(1,1,-1)
    for data_sp, targets in train_loader_sp:
        targets = targets.to(device, non_blocking=True)
        
        out_sp = model(data_sp, use_ema=use_ema)
        pred = (out_sp * prior).sum(dim=-1).argmax(dim=1)
        correct += (pred.flatten() == targets.flatten()).sum().item()
        n_total += len(targets)
        
        prob = out_sp[range(len(out_sp)), targets.flatten()]
        logLikelihood_sp = logLikelihood_sp + torch.log(prob).sum(dim=0)
        sp_cnt += len(data_sp)
    
    train_acc = correct / n_total
    
    logLikelihood_sp = logLikelihood_sp / sp_cnt
    Likelihood_sp = torch.exp(logLikelihood_sp)
    
    n_total = 0
    correct = 0
    prior = model.get_prior(use_ema=use_ema).reshape(1,1,-1) * Likelihood_sp.reshape(1,1,-1)
    for data_sp, targets in val_loader:
        targets = targets.to(device, non_blocking=True)
        out_sp = model(data_sp, use_ema=use_ema)
        
        prob_y = (out_sp * prior).sum(dim=-1)
        prob_y = prob_y / prob_y.sum(-1, keepdim=True)
        pred = prob_y.argmax(dim=1)
        correct += (pred.flatten() == targets.flatten()).sum().item()
        n_total += len(targets)
    
    bayes_val_acc = correct / n_total
    return train_acc, bayes_val_acc

def train(model, optimizer, scheduler, scaler, args,
        train_loader_sp, train_loader_unsp, val_loader):

    start_epoch = 0
    if args.reload_dir != "":
        start_epoch = int(args.reload_dir.split("epoch=")[-1])
    
    best_val_acc = 0.
    if start_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(args.reload_dir, "checkpoint.pth")))
        # for i in range(model.args.K):
        #     model.model.particles[i] = model.model.particles[i].to(model.model.devices[i], non_blocking=True)
        #     model.ema_model.particles[i] = model.ema_model.particles[i].to(model.model.devices[i], non_blocking=True)
        
        optimizer.load_state_dict(torch.load(os.path.join(args.reload_dir, "optimizer.pth")))
        scheduler.load_state_dict(torch.load(os.path.join(args.reload_dir, "scheduler.pth")))
        with open(os.path.join(args.reload_dir, "metrics.json"), "r") as f:
            metrics = json.load(f)
            best_val_acc = metrics["val"]["acc"][-1]
    
    e_loader_step = args.e_step
    report_freq = 100
    eval_freq = 1

    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    metrics = {"train": train_metrics, "val": val_metrics}

    # evaluation
    train_acc, bayes_val_acc = evaluate_bayes(model, train_loader_sp, val_loader)
    val_acc = evaluate(model, val_loader)
    val_acc_np = evaluate_noprior(model, val_loader)
    train_ema_acc, bayes_val_ema_acc = evaluate_bayes(model, train_loader_sp, val_loader, use_ema=True)
    val_ema_acc = evaluate(model, val_loader, use_ema=True)
    val_ema_acc_np = evaluate_noprior(model, val_loader, use_ema=True)
    
    print("Epoch {}/{}, lr:{:.4f}".format(0, args.epochs, scheduler.get_last_lr()[0]))
    print("      train acc: {:.4f}, val acc: {:.4f}, np val acc: {:.4f}, bayes val acc: {:.4f}".format(
        train_acc, val_acc, val_acc_np, bayes_val_acc))
    print("(ema) train acc: {:.4f}, val acc: {:.4f}, np val acc: {:.4f}, bayes val acc: {:.4f}".format(
        train_ema_acc, val_ema_acc, val_ema_acc_np, bayes_val_ema_acc))
    
    mi_run, ce_run, loss_run = 0.0, 0.0, 0.0
    H_y_run, H_yw_run = 0.0, 0.0
    entropy_run, mask_run = 0.0, 0.0
    t0 = time.time()
    for e in range(start_epoch, args.run_epochs):
        _ = model.train()

        sp_loader_iter = iter(train_loader_sp)
        unsp_loader_iter = iter(train_loader_unsp)

        # read data
        t1 = time.time()
        for step in range(e_loader_step):
            try:
                data_sp, targets = sp_loader_iter.next()
            except:
                sp_loader_iter = iter(train_loader_sp)
                data_sp, targets = sp_loader_iter.next()
                
            try:
                data_unsp, _ = unsp_loader_iter.next()
            except:
                unsp_loader_iter = iter(train_loader_unsp)
                data_unsp, _ = unsp_loader_iter.next()
            data_unsp_w, data_unsp_s = data_unsp
            
            if args.use_mixup:
                data_unsp_w, _ = mixup_batch(data_unsp_w, _, args)
                data_unsp_s, _ = mixup_batch(data_unsp_s, _, args)
            
            with torch.cuda.amp.autocast():
                ### deep prior MI ###
                out = model(torch.cat([data_sp, data_unsp_w, data_unsp_s]))
                out_sp, out_unsp = out[ : len(data_sp)], out[len(data_sp) : ]
                del out
                
                logLikelihood = model.get_logLikelihood(out_sp, targets)
                MI, H_y, H_yw, mask_ratio = model.get_MI(out_unsp)

                ### update ###
                optimizer.zero_grad()
                loss = -(args.gamma * MI + logLikelihood)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scale = scaler.get_scale()
                scaler.update()
                is_step_skipped = scale > scaler.get_scale()
            
            scheduler.step()
            model.update_ema()
            
            prior = model.get_prior().detach()
            entropy = -torch.sum(torch.log(prior) * prior)
            
            metrics["train"]["MI"].append(MI.item())
            metrics["train"]["ce_loss"].append(-logLikelihood.item())
            metrics["train"]["loss"].append(loss.item())
            metrics["train"]["entropy"].append(entropy.item())
            
            mi_run += MI.item()
            ce_run += (-logLikelihood.item())
            loss_run += loss.item()
            H_y_run += H_y.item()
            H_yw_run += H_yw.item()
            entropy_run += entropy.item()
            mask_run += mask_ratio.item()
            
            # verbose
            if (step+1) % report_freq == 0:
                print("Step {}/{}, lr:{:.8f}, loss: {:.4f}, MI: {:.4f}, ce_loss: {:.4f}, H_y: {:.4f}, H_yw: {:.4f}, info_loss: {:.4f}".format(
                    step+1, args.e_step, scheduler.get_last_lr()[0], loss.item(), MI.item(), -logLikelihood.item(), H_y.item(), H_yw.item(), (H_y-H_yw).item()))
        
        mi_run /= e_loader_step
        ce_run /= e_loader_step
        loss_run /= e_loader_step
        H_y_run /= e_loader_step
        H_yw_run /= e_loader_step
        entropy_run /= e_loader_step
        mask_run /= e_loader_step
        t2 = time.time()
        
        print("Epoch {}/{}, lr:{:.4f}, e_time: {:.2f} mins, total_time: {:.2f} mins".format(e+1, args.epochs, scheduler.get_last_lr()[0], (t2 - t1)/60, (t2 - t0)/60))
        print("(accumu) mi:{:.4f}, ce_loss:{:.4f}, loss:{:.4f}, H_y:{:.4f}, H_yw:{:.4f}, entropy:{:.4f}, mask_ratio:{:.4f}".format(
            mi_run, ce_run, loss_run, H_y_run, H_yw_run, entropy_run, mask_run))

        # evaluation
        if (e+1) % eval_freq == 0:
            train_acc, bayes_val_acc = evaluate_bayes(model, train_loader_sp, val_loader)
            val_acc = evaluate(model, val_loader)
            val_acc_np = evaluate_noprior(model, val_loader)
            train_ema_acc, bayes_val_ema_acc = evaluate_bayes(model, train_loader_sp, val_loader, use_ema=True)
            val_ema_acc = evaluate(model, val_loader, use_ema=True)
            val_ema_acc_np = evaluate_noprior(model, val_loader, use_ema=True)
            
            # verbose
            print("      train acc: {:.4f}, val acc: {:.4f}, np val acc: {:.4f}, bayes val acc: {:.4f}".format(
                train_acc, val_acc, val_acc_np, bayes_val_acc))
            print("(ema) train acc: {:.4f}, val acc: {:.4f}, np val acc: {:.4f}, bayes val acc: {:.4f}".format(
                train_ema_acc, val_ema_acc, val_ema_acc_np, bayes_val_ema_acc))
            
            metrics["train"]["acc"].append(train_acc)
            metrics["train"]["ema_acc"].append(train_ema_acc)
            metrics["val"]["acc"].append(val_acc)
            metrics["val"]["acc_np"].append(val_acc_np)
            metrics["val"]["ema_acc"].append(val_ema_acc)
            metrics["val"]["ema_acc_np"].append(val_ema_acc_np)
            metrics["val"]["bayes_acc"].append(bayes_val_acc)
            metrics["val"]["bayes_ema_acc"].append(bayes_val_ema_acc)
            
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_results(model, optimizer, scheduler, metrics, e+1, args)
        
    return model, metrics