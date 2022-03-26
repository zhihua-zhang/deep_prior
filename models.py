import itertools
from copy import deepcopy

import torch
from torch import nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

from utils import weak_augmentation

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class BaseModel_fc(nn.Module):
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

class BaseModel_cnn(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        # input img: (1, 28, 28)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (5, 5), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (5, 5), padding="same", bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, (5, 5), padding="same", bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool2d((2, 2), stride=2, padding=0)
        self.rep = nn.Sequential(
            self.conv1,
            self.avg_pool,
            self.conv2,
            self.avg_pool,
            self.conv3,
            self.avg_pool,
            nn.Flatten()
        )
        self.clf = nn.Sequential(
            nn.Linear(576, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.rep(x)
        out = self.clf(x)
        return out

def build_BaseModel(in_dim, out_dim, hidden_dims, args):
    if args.base_model == 'fc':
        base_model = BaseModel_fc(in_dim, out_dim, hidden_dims)
    elif args.base_model == 'cnn':
        base_model = BaseModel_cnn(out_dim)
    else:
        raise Exception("Wrong base model type.")
    return base_model

class DeepPrior(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, args):
        super().__init__()
        self.deep_priors = nn.ModuleList([build_BaseModel(in_dim, out_dim, hidden_dims, args) for _ in range(args.K)])
        self.n_order = args.n_order
        self.K = args.K
        self.alpha = args.alpha
        self.args = args
    
    def forward(self, x):
        outs = []
        for prior_model in self.deep_priors:
            out = prior_model(x).unsqueeze(-1)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs
    
    def get_MI(self, out_unsp):
        """
        Input
            out_unsp: (bz,n_label,K)

            For each data pair, there are n_label^n_order combinations in total,
        so we use itertools to obtain per-particle cross product.
        """

        bz, n_label, K = out_unsp.size()
        out_unsp = out_unsp.reshape(self.n_order, bz//self.n_order, n_label, K)
        out_unsp = out_unsp.permute(0, 2, 3, 1)
        if self.args.target_comb == "cross_product":
            indices = torch.tensor([[d1, d2] for d1 in range(self.n_order) for d2 in range(n_label)])
            # combinations = list(itertools.product(*out_unsp))
        elif self.args.target_comb == "permutations":
            indices = list(itertools.permutations(range(n_label), r=self.n_order))
            # combinations = [[out_unsp[i][target] for i, target in enumerate(permute)] for permute in permutations]
        
        prob_all = []
        for idx in indices:
            pair = out_unsp[range(self.n_order), idx]
            # prob = torch.stack(pair).prod(dim=0)
            prob = pair.prod(dim=0)
            prob_all.append(prob)
        del out_unsp
        
        # prob_all: (n_label^n_order, K, bz//n_order)
        prob_all = torch.stack(prob_all) + 1e-7
        
        # prob_per_y: (n_label^n_order, bz//n_order)
        prob_per_y = prob_all.mean(dim=1)
        H_y = (-prob_per_y * torch.log(prob_per_y)).sum()

        H_yw = (-prob_all * torch.log(prob_all)).sum(dim=0).mean(dim=0).sum()
        
        MI = self.alpha * H_y - H_yw
        return MI

    def get_logLikelihood(self, out_sp, targets):
        """
        out_unsp: (bz,n_label,K)
        """
        prob = out_sp[range(len(out_sp)), targets.flatten()]
        logLikelihood = torch.mean(torch.log(prob).sum(dim=0))
        return logLikelihood


class ModelEma(DeepPrior):
    def __init__(self, model, in_dim, hidden_dims, out_dim, args, decay=0.999):
        super().__init__(in_dim, hidden_dims, out_dim, args)
        # make a copy of the model for accumulating moving average of weights
        # only use for evaluation
        # self = deepcopy(model)
        self.eval()
        self.decay = decay
        self.set(model)
        
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
    
    def forward(self, x):
        outs = []
        for prior_model in self.deep_priors:
            out = prior_model(x).unsqueeze(-1)
            outs.append(out)
        outs = torch.cat(outs, dim=-1)
        return outs