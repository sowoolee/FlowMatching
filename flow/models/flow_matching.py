import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torchdiffeq import odeint
from functools import partial
from .helpers import apply_conditioning

class CondOTFlowMatching(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim,
        hidden_dim=256,
        condition_guidance_w=0.1):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        self.inv_model = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )
        self.condition_guidance_w = condition_guidance_w

    #------------------------------------------ training ------------------------------------------#
    def loss(self, x, cond, returns=None):
        # flow matching loss
        batch_size = len(x)
        t = torch.rand(batch_size, device=x.device)
        flow_loss = self.flow_loss(x[:, :, self.action_dim:], cond, t, returns)

        # inverse dynamics loss
        x_t = x[:, :-1, self.action_dim:]
        a_t = x[:, :-1, :self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        a_t = a_t.reshape(-1, self.action_dim)

        pred_a_t = self.inv_model(x_comb_t)
        inv_loss = F.mse_loss(pred_a_t, a_t)

        loss = (1/2) * (flow_loss + inv_loss)

        return loss

    def flow_loss(self, x1, cond, t, returns=None):
        x0 = torch.rand_like(x1)

        # OT Path
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        xt = xt.to(x1.device)
        # xt = apply_conditioning(xt, cond, 0)
        true_flow = x1 - x0

        pred_flow = self.model(xt, cond, t, returns)

        loss = F.mse_loss(true_flow, pred_flow)

        return loss

    def loss_debug(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.zeros(batch_size, device=x.device)
        x1 = x[:, :, self.action_dim:]
        x0 = torch.rand_like(x1)

        xt = apply_conditioning(x0, cond, 0)
        pred_flow = self.model(xt, cond, t, returns)
        pred_x1 = x0 + pred_flow

        x_t = x[:, :-1, self.action_dim:]
        a_t = x[:, :-1, :self.action_dim]
        x_t_1 = x[:, 1:, self.action_dim:]
        x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
        x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
        a_t = a_t.reshape(-1, self.action_dim)
        pred_a_t = self.inv_model(x_comb_t)

        return x1, pred_x1, a_t, pred_a_t

    #------------------------------------------ sampling ------------------------------------------#
    def inference(self, cond, returns=None):
        device = returns.device
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        x0 = torch.rand(batch_size, horizon, self.observation_dim).to(device)
        # x0 = apply_conditioning(x0, cond, 0)
        t = torch.tensor([0.0, 1.0]).to(device)

        # v_fn = partial(self.model, cond=cond, returns=returns)
        v_fn = lambda t, xt: self.model(xt, cond, t.unsqueeze(-1), returns)

        with torch.no_grad():
            x1 = odeint(v_fn, x0, t, method='midpoint', atol=1e-5, rtol=1e-5)
        return x1[-1]

    def inference_1step(self, cond, returns=None):
        device = returns.device
        batch_size = len(cond[0])
        t = torch.zeros(batch_size, device=device)
        horizon = self.horizon
        x0 = torch.rand(batch_size, horizon, self.observation_dim).to(device)

        xt = apply_conditioning(x0, cond, 0)
        pred_flow = self.model(xt, cond, t, returns, False)
        return x0 + pred_flow

    def forward(self, cond, *args, **kwargs):
        return self.inference(cond=cond, *args, **kwargs)