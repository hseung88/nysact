from typing import List
import torch
import torch.nn as nn
from torch.optim import Optimizer
import logging as log
from .utils.opt_utils import extract_patches, reshape_grad, momentum_step
from .utils.torch_utils import build_layer_map


class FOOF(Optimizer):
    def __init__(self,
                 params,
                 lr=0.1,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=1.0,
                 weight_decay=1e-5,
                 Tcov=5,
                 Tinv=50):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if Tcov > Tinv:
            raise ValueError("Tcov={Tcov:d} < Tinv={Tinv:d}")

        defaults = dict(lr=lr,
                        momentum=momentum,
                        stat_decay=stat_decay,
                        damping=damping,
                        weight_decay=weight_decay)
        super(FOOF, self).__init__(params, defaults)

        self._model = None
        self._step = 0
        self.emastep = 0
        self.stat_decay = stat_decay
        self.Tcov = Tcov
        self.Tinv = Tinv

    @property
    def model(self):
        if self._model is None:
            log.error("Model is not attached to the optimizer.")

        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self.layer_map = build_layer_map(model, fwd_hook_fn=self._capture_activation)

    def _capture_activation(
            self,
            module: nn.Module,
            forward_input: List[torch.Tensor],
            _forward_output: torch.Tensor
    ):
        if not module.training or not torch.is_grad_enabled():
            return

        if self._step % self.Tcov != 0:
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

        A = actv.t() @ (actv / actv.size(0))

        state = self.state[module]
        if 'ema_A' not in state:
            state['ema_A'] = torch.eye(A.size(0), device=A.device)

        state['ema_A'].mul_(1 - stat_decay).add_(A, alpha=stat_decay)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        stat_decay = group['stat_decay']
        damping = group['damping']

        b_inv_update = ((self._step % self.Tinv) == 0)

        # compute the preconditioned gradient layer-by-layer
        for layer in self.layer_map:
            if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.grad is not None:
                state = self.state[layer]
                grad_mat = reshape_grad(layer)

                if b_inv_update:
                    bias_correction = 1.0 - (stat_decay ** self.emastep)
                    ema_A = state['ema_A'].div(bias_correction)

                    if 'A_inv' not in state:
                        state['A_inv'] = torch.eye(ema_A.size(0), device=ema_A.device)
                    else:
                        eye_matrix = torch.eye(ema_A.size(0), device=ema_A.device)
                        state['A_inv'] = torch.cholesky_inverse(torch.linalg.cholesky(ema_A + damping * eye_matrix))

                A_inv = state['A_inv']

                v = grad_mat @ A_inv

                if layer.bias is not None:
                    v = [v[:, :-1], v[:, -1:]]
                    v[0] = v[0].view_as(layer.weight)
                    v[1] = v[1].view_as(layer.bias)

                    layer.weight.grad.data.copy_(v[0])
                    layer.bias.grad.data.copy_(v[1])
                else:
                    v = v.view(layer.weight.grad.size())
                    layer.weight.grad.data.copy_(v)

        momentum_step(self)
        self._step += 1

        return loss
