from ast import Or
import itertools
from copy import deepcopy
from functools import partial
from collections import OrderedDict
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        args,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bn1 = norm_layer(inplanes, momentum=args.bn_momentum)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = norm_layer(planes, momentum=args.bn_momentum)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, activate_before_residual=False):
        out = self.bn1(x)
        out = self.leaky_relu(out)
        
        if activate_before_residual:
            identity = out
        else:
            identity = x
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_classes,
        layers,
        args,
        widen_factor=2,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        # _log_api_usage_once(self)
        
        self.args = args
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding="same", bias=False)
        layer1 = self._make_layer(block, 16*widen_factor, layers[0])
        layer2 = self._make_layer(block, 32*widen_factor, layers[1], stride=2)
        layer3 = self._make_layer(block, 64*widen_factor, layers[2], stride=2)
        bn1 = norm_layer(64*widen_factor, momentum=args.bn_momentum)
        
        self.feat = nn.Sequential(
            conv1,
            layer1,
            layer2,
            layer3,
            bn1,
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten = nn.Flatten()

        self.clf = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                # norm_layer(planes),
                # norm_layer(planes, momentum=self.args.bn_momentum),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, self.args, stride, downsample, norm_layer
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    self.args,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor):
        x = self.feat(x)
        x = self.flatten(x)
        x = self.clf(x)
        
        return x

def build_BaseModel(num_classes, layers, args):
    model = ResNet(BasicBlock, num_classes, layers, args=args)
    return model

class DeepPrior(nn.Module):
    def __init__(self, num_classes, n_blocks, args):
        super().__init__()
        self.args = args
        self.particles = nn.ModuleList([build_BaseModel(num_classes, n_blocks, args) for _ in range(args.K)])
        self.devices = []
        for i in range(args.K):
            d = (args.main_device + i) % 4
            while d in args.skip_devices:
                d = (d+1) % 4
            device = f"cuda:{d}" if torch.cuda.is_available() else "cpu"
            self.particles[i] = self.particles[i].to(device)
            self.particles[i].device = device
            self.devices.append(device)
        
        d = args.main_device
        while d in args.skip_devices:
            d = (d+1) % 4
        self.main_device = f"cuda:{d}" if torch.cuda.is_available() else "cpu"
        
        self.logprior = torch.tensor([0.] * self.args.K, device=self.main_device)
        if args.train_prior:
            self.logprior = nn.Parameter(self.logprior)
    
    def forward(self, x):
        outs = [net(x.to(self.devices[i])) for i, net in enumerate(self.particles)]
        outs = [pred.to(self.main_device) for pred in outs]
        outs = torch.stack(outs, dim=2)
        return outs
    
    def get_prior(self):
        prior = F.softmax(self.logprior, dim=0)
        return prior
    
    def get_all_prob(self, out_unsp):
        """
        Input
            out_unsp: (2*bz,n_label,K)
        Output
            prob_all: (2*bz//n_order, K, n_label^n_order)
        """
        bz, n_label, K = out_unsp.size()
        out_unsp = out_unsp.permute(0, 2, 1)
        
        bz_order = bz//self.args.n_order
        out_unsp = out_unsp.reshape(self.args.n_order, bz_order, K, n_label)
        
        out_unsp_all = []
        for i in range(self.args.n_order):
            reshape_i = [bz_order, K] + [1] * self.args.n_order
            reshape_i[2+i] = n_label
            out_unsp_i = out_unsp[i].reshape(reshape_i)
            out_unsp_all.append(torch.log(out_unsp_i))
        
        prob_all = torch.exp(sum(out_unsp_all)) # shape: (2*bz//n_order, K, n_label, ... , n_label)
        prob_all = prob_all.reshape(bz_order, K, -1) # shape: (2*bz//n_order, K, n_label^n_order)
        return prob_all
    
    def get_MI(self, out_unsp):
        """
        Input
            out_unsp: (2*bz,n_label,K)
        """

        # shape: (bz//n_order, K, n_label^n_order)
        prob_all_w, prob_all_s = self.get_all_prob(out_unsp).chunk(2, dim=0)
        
        ## calc H_y
        prior = self.get_prior().reshape(1,-1,1)
        if self.args.Hy_use_weak:
            prob_per_y = (prob_all_w * prior).sum(dim=1)
        else:
            prob_all_avg = self.args.tau * prob_all_w + (1 - self.args.tau) * prob_all_s
            prob_per_y = (prob_all_avg * prior).sum(dim=1)
        H_y = (-prob_per_y * torch.log(prob_per_y + 1e-12)).sum(dim=1).mean() / self.args.n_order

        ## calc H_yw
        out_unsp_w, out_unsp_s = out_unsp.chunk(2, dim=0)
        if self.args.no_gradient_stop:
            p_w = out_unsp_w
        else:
            p_w = out_unsp_w.detach()
        p_s = out_unsp_s
        
        if self.args.no_jensen:
            p_avg = (self.args.tau * p_w + (1 - self.args.tau) * p_s)
            entropy = (-p_avg * torch.log(p_avg + 1e-12)).sum(dim=1)
        else:
            p_avg = (self.args.tau * p_w + (1 - self.args.tau) * p_s)
            entropy = (-p_avg * (self.args.tau * torch.log(p_w + 1e-12) + (1 - self.args.tau) * torch.log(p_s + 1e-12))).sum(dim=1)
        
        mask = p_w.max(dim=1)[0].ge(self.args.label_thresh)
        prior = prior.reshape(1,-1)
        H_yw = (entropy * prior * mask).sum(dim=1).mean()
        
        # output
        with torch.no_grad():
            p_w, p_s = out_unsp.chunk(2, dim=0)
            
            pw_prob = p_w.max(dim=1)[0]
            ps_prob = p_s.max(dim=1)[0]
            pmax_stat = torch.tensor([pw_prob.mean(), pw_prob.std(), ps_prob.mean(), ps_prob.std()])
        mask_ratio = mask.sum() / mask.nelement()
        MI = self.args.alpha * H_y - H_yw

        return MI, H_y, H_yw, mask_ratio, pmax_stat
    
    def get_logLikelihood(self, out_sp, targets):
        """
        out_unsp: (bz,n_label,K)
        """
        prob = out_sp[range(len(out_sp)), targets.flatten()]
        logLikelihood = torch.log(prob)
        prior = self.get_prior().reshape(1,-1)
        logLikelihood = (logLikelihood * prior).sum(dim=1).mean()
        return logLikelihood


class EmaModel(nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        # only use for evaluation
        self.model = model
        self.decay = decay
        self.args = self.model.args
        self.main_device = self.model.main_device
        self.ema_model = deepcopy(model)
        
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model.eval()
        
    @torch.no_grad()
    def update_ema(self):
        ema_model_params = OrderedDict(self.ema_model.named_parameters())
        model_params = OrderedDict(self.model.named_parameters())
        for name in ema_model_params:
            ema_model_params[name].add_((1. - self.decay) * (model_params[name] - ema_model_params[name]))

        ema_model_buffers = OrderedDict(self.ema_model.named_buffers())
        model_buffers = OrderedDict(self.model.named_buffers())
        for name, buff in model_buffers.items():
            ema_model_buffers[name].copy_(buff)
            
    def forward(self, x, use_ema=False):
        if use_ema:
            return self.ema_model(x)
        else:
            return self.model(x)
            
    def get_opt_params(self, wd):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'logprior' in name or ".bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': wd}]
    
    def get_prior(self, use_ema=False):
        if use_ema:
            return self.ema_model.get_prior()
        else:
            return self.model.get_prior()
    
    def get_MI(self, out_unsp):
        return self.model.get_MI(out_unsp)
    
    def get_logLikelihood(self, out_sp, targets):
        return self.model.get_logLikelihood(out_sp, targets)