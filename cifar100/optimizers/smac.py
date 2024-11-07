from typing import List
import logging as log
import torch
import torch.nn as nn
from torch.optim import Optimizer
from .utils.mac_utils import extract_patches, reshape_grad, build_layer_map, trainable_modules, momentum_step, nag_step

def try_contiguous(x):
    return x if x.is_contiguous() else x.contiguous()

class SMAC(Optimizer):
    def __init__(
        self,
        params,
        lr=0.1,
        momentum=0.9,
        stat_decay=0.95,
        damping=1.0,
        weight_decay=5e-4,
        Tcov=5,
        Tinv=50,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if stat_decay < 0.0:
            raise ValueError(f"Invalid stat_decay value: {stat_decay}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, 
                        momentum=momentum, 
                        stat_decay=stat_decay,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        self._model = None
        self.damping = damping
        self.Tcov = Tcov
        self.Tinv = Tinv
        self._step = 0
        self.emastep = 0

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model,
                                         fwd_hook_fn=self._capture_activation,
                                         bwd_hook_fn=self._capture_backprop)


    def _capture_activation(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor
    ):
        if not module.training or not torch.is_grad_enabled():
            return
            
        if (self._step % self.Tcov) != 0:
            return

        self.emastep += 1
            
        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        actv = forward_input[0].data
        if isinstance(module, nn.Conv2d):
            depthwise = module.groups == actv.size(1)
            actv = extract_patches(actv, module.kernel_size, module.stride, module.padding, depthwise)
        elif isinstance(module, nn.Linear):
            if actv.ndim > 2:  # linear layers in transformers
                actv = actv.view(-1, actv.size(-1))

        if module.bias is not None:
            ones = torch.ones((actv.size(0), 1), device=actv.device, dtype=actv.dtype)
            actv = torch.cat([actv, ones], dim=1)

        avg_actv = actv.mean(0)

        state = self.state[module]
        if 'exp_avg' not in state:
            state['exp_avg'] = torch.zeros_like(avg_actv)
        state['exp_avg'].mul_(stat_decay).add_(avg_actv, alpha=1 - stat_decay)

    def _capture_backprop(
        self,
        module: nn.Module,
        _grad_input: torch.Tensor,
        grad_output: torch.Tensor
    ):        
        if (self._step % self.Tcov) != 0:
            return
            
        group = self.param_groups[0]
        stat_decay = group['stat_decay']

        g = grad_output[0].data
        if isinstance(module, nn.Conv2d):
            spatial_size = g.size(2) * g.size(3)
            g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))
        #if isinstance(module, nn.Conv2d):
        #    g *= spatial_size

        avg_dz = g.pow(2).mean(0)
        state = self.state[module]
        if 'exp_avg_z' not in state:
            state['exp_avg_z'] = torch.zeros_like(avg_dz, device=avg_dz.device)

        state['exp_avg_z'].mul_(stat_decay).add_(avg_dz, alpha=1 - stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = self.damping
        b_updated = (self._step % self.Tinv == 0)


        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if b_updated:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    exp_avg_z = state['exp_avg_z'].div(bias_correction)
                    if 'Z_inv' not in state:
                        state['Z_inv'] = torch.eye(exp_avg_z.size(0), device=exp_avg_z.device)
                    
                    Z = exp_avg_z.add_(damping).reciprocal_()
                    state['Z_inv'].diagonal().copy_(Z)
               
                    exp_avg = state['exp_avg'].div(bias_correction)
                    sq_norm = torch.linalg.norm(exp_avg).pow(2)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(exp_avg.size(0), device=exp_avg.device)
                    else:
                        state['A_inv'].copy_(torch.eye(exp_avg.size(0), device=exp_avg.device))

                    state['A_inv'].sub_(torch.outer(exp_avg, exp_avg).div_(damping + sq_norm))
                    state['A_inv'].div_(damping)
                    
                Z_inv = state['Z_inv']
                A_inv = state['A_inv']

                v = Z_inv @ grad_mat @ A_inv

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    layer.weight.grad.data.copy_(v[0].view_as(layer.weight))
                    layer.bias.grad.data.copy_(v[1].view_as(layer.bias))
                else:
                    layer.weight.grad.data.copy_(v.view_as(layer.weight.grad))

        momentum_step(self)
        self._step += 1

        return loss
