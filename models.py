from audioop import cross
from turtle import forward
import itertools
from copy import deepcopy

import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from utils import weak_augmentation

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BaseModel(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[32]):
        super().__init__()

        hidden_layers = []
        for i in range(len(hidden_dims)):
            if i == 0:
                hidden_layers.append(nn.Linear(in_dim, hidden_dims[i]))
            else:
                hidden_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            # hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Tanh())
        
        self.rep = nn.Sequential(*hidden_layers)
        self.clf = nn.Sequential(
            nn.Linear(hidden_dims[-1], out_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.rep(x)
        out = self.clf(x)
        return out


class DeepPrior(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, n_order, K, alpha=1.0):
        super().__init__()
        self.augment = weak_augmentation()
        
        self.deep_priors = nn.ModuleList([BaseModel(in_dim, hidden_dims, out_dim) for _ in range(K)])
        self.n_order = n_order
        self.K = K
        self.alpha = alpha
    
    def forward(self, x):
        outs = []
        for prior_model in self.deep_priors:
            out = prior_model(x).unsqueeze(-1)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs
    
    def get_MI_single(self, out_unsp_i):
        """
        Input
            out_unsp_i: (n_order,n_target, K)

            Note, there are n_target^n_order combinations in total,
        so we use itertools to obtain per-particle cross product.
        """
        
        cross_prod = list(itertools.product(*out_unsp_i))
        prob_all = []
        for pair in cross_prod:
            prob = torch.stack(pair).prod(dim=0)
            prob_all.append(prob)
        prob_all = torch.stack(prob_all)
        
        # probs: (n_target^n_order, K)
        prob_per_pred = torch.mean(prob_all, dim=-1)
        pred_entropy_i = torch.sum(-prob_per_pred * torch.log(prob_per_pred))

        particle_entropy_i = torch.mean(torch.sum(-prob_all * torch.log(prob_all), dim=0))
        return pred_entropy_i, particle_entropy_i

    def get_MI(self, out_unsp):
        """
        out_unsp: (bz,n_target,K)
        """
        bz = len(out_unsp)

        pred_entropy = torch.zeros(1, device=device)
        particle_entropy = torch.zeros(1, device=device)
        for i in range(0, bz, self.n_order):
            out_unsp_i = out_unsp[i : i+self.n_order]
            pred_entropy_i, particle_entropy_i = self.get_MI_single(out_unsp_i)

            pred_entropy = pred_entropy + pred_entropy_i
            particle_entropy = particle_entropy + particle_entropy_i
        
        MI = self.alpha * pred_entropy - particle_entropy
        return MI

    def get_logLikelihood(self, out_sp, targets):
        """
        out_unsp: (bz,n_target,K)
        """
        prob = out_sp[range(len(out_sp)), targets.flatten()]
        logLikelihood = torch.mean(torch.log(prob.prod(dim=0)))
        return logLikelihood


class ModelEma(nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        # only use for evaluation
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)