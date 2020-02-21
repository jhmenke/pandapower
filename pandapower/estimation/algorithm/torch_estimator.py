# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import warnings
from functools import partial

import numpy as np
import torch
import torch.jit
from torch import tensor as t, double as td
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pandapower.estimation.algorithm.base import WLSAlgorithm
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE
from pandapower.pypower.makeYbus import makeYbus


@torch.jit.script
def real_matmul(m1r, m1i, m2r, m2i):
    return m1r @ m2r - m1i @ m2i


@torch.jit.script
def imag_matmul(m1r, m1i, m2r, m2i):
    return m1r @ m2i + m1i @ m2r


@torch.jit.script
def real_mul(m1r, m1i, m2r, m2i):
    return m1r * m2r - m1i * m2i


@torch.jit.script
def imag_mul(m1r, m1i, m2r, m2i):
    return m1r * m2i + m1i * m2r


class TorchEstimator(torch.nn.Module):
    def __init__(self, eppci):
        super(TorchEstimator, self).__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Ybus, Yf, Yt = makeYbus(eppci["baseMVA"], eppci["bus"], eppci["branch"])
            eppci['internal']['Yf'], eppci['internal']['Yt'], \
            eppci['internal']['Ybus'] = Yf, Yt, Ybus

        Ybus, Yf, Yt = Ybus.toarray(), Yf.toarray(), Yt.toarray()
        self.yfr = torch.from_numpy(np.real(Yf).astype(np.double))
        self.yfi = torch.from_numpy(np.imag(Yf).astype(np.double))
        self.ytr = torch.from_numpy(np.real(Yt).astype(np.double))
        self.yti = torch.from_numpy(np.imag(Yt).astype(np.double))
        self.ybr = torch.from_numpy(np.real(Ybus).astype(np.double))
        self.ybi = torch.from_numpy(np.imag(Ybus).astype(np.double))

        self.slack_bus = np.argwhere(eppci["bus"][:, BUS_TYPE] == 3).ravel()
        self.non_slack_bus = np.argwhere(eppci["bus"][:, BUS_TYPE] != 3).ravel()
        self.fbus = torch.from_numpy(np.abs(eppci["branch"][:, F_BUS])).long()
        self.tbus = torch.from_numpy(np.abs(eppci["branch"][:, T_BUS])).long()

        # ignore current measurement
        non_nan_meas_mask = eppci.non_nan_meas_mask[:3 * len(eppci["bus"]) + 4 * len(eppci["branch"])]
        self.non_nan_meas_mask = t(non_nan_meas_mask).bool().view(-1, 1)
        self.vi_slack = torch.zeros(self.slack_bus.shape[0], 1, requires_grad=False, dtype=td)
        self.vi_mapping = torch.from_numpy(np.argsort(np.r_[self.slack_bus, self.non_slack_bus])).long()

    def forward(self, vr, vi_non_slack):
        vi = torch.cat((self.vi_slack, vi_non_slack), 0)
        vi = vi.index_select(0, self.vi_mapping)

        p_b = real_mul(vr, vi, real_matmul(self.ybr, self.ybi, vr, vi),
                       - imag_matmul(self.ybr, self.ybi, vr, vi))
        q_b = imag_mul(vr, vi, real_matmul(self.ybr, self.ybi, vr, vi),
                       - imag_matmul(self.ybr, self.ybi, vr, vi))

        vfr, vfi = torch.index_select(vr, 0, self.fbus), torch.index_select(vi, 0, self.fbus)
        p_f = real_mul(vfr, vfi, real_matmul(self.yfr, self.yfi, vr, vi),
                       - imag_matmul(self.yfr, self.yfi, vr, vi))
        q_f = imag_mul(vfr, vfi, real_matmul(self.yfr, self.yfi, vr, vi),
                       - imag_matmul(self.yfr, self.yfi, vr, vi))

        vtr, vti = torch.index_select(vr, 0, self.tbus), torch.index_select(vi, 0, self.tbus)
        p_t = real_mul(vtr, vti, real_matmul(self.ytr, self.yti, vr, vi),
                       - imag_matmul(self.ytr, self.yti, vr, vi))
        q_t = imag_mul(vtr, vti, real_matmul(self.ytr, self.yti, vr, vi),
                       - imag_matmul(self.ytr, self.yti, vr, vi))

        hx_pq = torch.cat([p_b, p_f, p_t, q_b, q_f, q_t], 0)
        hx_v = torch.sqrt(vr ** 2 + vi ** 2)
        hx = torch.masked_select(torch.cat([hx_pq, hx_v]), self.non_nan_meas_mask)
        return hx


@torch.jit.script
def weighted_mse_loss(predicted, target, weight):
    return torch.sum(((predicted - target) / weight)**2)


def optimize(model, floss, vr, vi_non_slack, optimizer="lbfgs"):
    if optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS([vr, vi_non_slack], lr=1.0, max_iter=100)
        max_epochs = 1
    else:
        optimizer = torch.optim.Adam([vr, vi_non_slack], lr=1.0)
        max_epochs = 1000
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history = []
    for e in range(max_epochs):
        def closure():
            optimizer.zero_grad()
            hx = model(vr, vi_non_slack)
            loss = floss(hx)
            # print(f"Epoch #{e} -> {loss.item()}")
            loss.backward()
            scheduler.step(loss, epoch=e)
            history.append(loss.item())
            return loss
        optimizer.step(closure)
        if e > 0 and abs(history[-2] - history[-1]) < 1e-6:
            break
    return vr, vi_non_slack, len(history) != max_epochs


class TorchAlgorithm(WLSAlgorithm):
    def estimate(self, eppci, opt_vars=None):
        self.initialize(eppci)
        model = TorchEstimator(eppci)
        floss = partial(weighted_mse_loss, target=t(eppci.z).double(), weight=t(eppci.r_cov).double())
        vr = t(eppci.E[len(eppci.non_slack_buses):][:, np.newaxis], dtype=td, requires_grad=True)
        # vi_non_slack = (torch.tan(t(eppci.E[:len(eppci.non_slack_buses), np.newaxis])) * vr[eppci.non_slack_buses, :]).clone().detach().double().requires_grad_(True)
        vi_non_slack = t(np.tan(eppci.E[:len(eppci.non_slack_buses), np.newaxis])
                         * eppci.E[len(eppci.non_slack_buses) + eppci.non_slack_buses][:, np.newaxis], dtype=td, requires_grad=True)
        vr, vi_non_slack, self.successful = optimize(model, floss, vr, vi_non_slack)
        vr = vr.detach().numpy().ravel()
        vi = np.zeros(len(eppci["bus"]))
        vi[model.non_slack_bus] = vi_non_slack.detach().numpy().ravel()
        eppci.update_E(np.concatenate((np.arctan(vi / vr)[model.non_slack_bus], np.sqrt(vr**2 + vi**2))))
        return eppci

