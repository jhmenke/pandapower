import warnings

import jax.numpy as np
import numpy as onp
import numpyro
import numpyro.distributions as dist
from jax import jit
from jax import random
from jax.numpy import array as a
from numpyro.distributions.constraints import interval, greater_than
from numpyro.infer import MCMC, NUTS

from pandapower.estimation.algorithm.base import WLSAlgorithm
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_TYPE
from pandapower.pypower.makeYbus import makeYbus


@jit
def p_i(i: int, v, delta, ybr, ybi):
    res = 0.
    for j in range(len(v)):
        res = np.add(res, v[j] * (ybr[i, j] * np.cos(delta[i] - delta[j]) + ybi[i, j] * np.sin(delta[i] - delta[j])))
    return v[i] * res


@jit
def q_i(i: int, v, delta, ybr, ybi):
    res = 0.
    for j in range(len(v)):
        res = np.add(res, v[j] * (ybr[i, j] * np.sin(delta[i] - delta[j]) - ybi[i, j] * np.cos(delta[i] - delta[j])))
    return v[i] * res


@jit
def p_ij(i: int, j: int, v, delta, yfr, yfi):
    return v[i] * v[i] * (yfr[i]) - v[i] * v[j] * (yfr[i] * np.cos(delta[i] - delta[j]) + yfi[i] * np.sin(delta[i] - delta[j]))


@jit
def q_ij(i: int, j: int, v, delta, yfr, yfi):
    return -v[i] * v[i] * (yfi[i]) - v[i] * v[j] * (yfr[i] * np.sin(delta[i] - delta[j]) - yfi[i] * np.cos(delta[i] - delta[j]))


@jit
def p_ji(i: int, j: int, v, delta, ytr, yti):
    return v[j] * v[j] * (ytr[j]) - v[j] * v[i] * (ytr[j] * np.cos(delta[j] - delta[i]) + yti[j] * np.sin(delta[j] - delta[i]))


@jit
def q_ji(i: int, j: int, v, delta, ytr, yti):
    return -v[j] * v[j] * (yti[j]) - v[j] * v[i] * (ytr[j] * np.sin(delta[j] - delta[i]) - yti[j] * np.cos(delta[j] - delta[i]))


class Hx:
    def __init__(self, eppci):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ybus, yf, yt = makeYbus(eppci["baseMVA"], eppci["bus"], eppci["branch"])
            eppci['internal']['Yf'], eppci['internal']['Yt'], eppci['internal']['Ybus'] = yf, yt, ybus
        ybus, yf, yt = ybus.toarray(), yf.toarray(), yt.toarray()
        self.yfr = np.real(yf.ravel())
        self.yfi = np.imag(yf.ravel())
        self.ytr = np.real(yt.ravel())
        self.yti = np.imag(yt.ravel())
        self.ybr = np.real(ybus)
        self.ybi = np.imag(ybus)
        self.slack_bus = onp.argwhere(eppci["bus"][:, BUS_TYPE] == 3).ravel()
        self.non_slack_bus = onp.argwhere(eppci["bus"][:, BUS_TYPE] != 3).ravel()
        self.fbus = np.abs(eppci["branch"][:, F_BUS].real.astype(int))
        self.tbus = np.abs(eppci["branch"][:, T_BUS].real.astype(int))
        self.bus_idx = np.arange(eppci["bus"].shape[0])
        self.m = eppci.non_nan_meas_mask
        self.bus_p_idx = range(0, len(eppci["bus"]))
        self.line_p_from_idx = range(len(eppci["bus"]), len(eppci["bus"]) + len(eppci["branch"]))
        self.line_p_to_idx = range(len(eppci["bus"]) + len(eppci["branch"]), len(eppci["bus"]) + 2 * len(eppci["branch"]))
        self.bus_q_idx = range(len(eppci["bus"]) + 2 * len(eppci["branch"]), 2 * len(eppci["bus"]) + 2 * len(eppci["branch"]))
        self.line_q_from_idx = range(2 * len(eppci["bus"]) + 2 * len(eppci["branch"]), 2 * len(eppci["bus"]) + 3 * len(eppci["branch"]))
        self.line_q_to_idx = range(2 * len(eppci["bus"]) + 3 * len(eppci["branch"]), 2 * len(eppci["bus"]) + 4 * len(eppci["branch"]))
        self.bus_v_idx = range(2 * len(eppci["bus"]) + 4 * len(eppci["branch"]), 3 * len(eppci["bus"]) + 4 * len(eppci["branch"]))
        # todo I measurements
        self.vi_slack = a(eppci.V[self.slack_bus].imag)
        self.eppci = eppci

    def __call__(self, v, delta):
        p_bus = a([p_i(bus, v, delta, self.ybr, self.ybi) for bus in self.bus_idx[self.m[self.bus_p_idx]]])
        q_bus = a([q_i(bus, v, delta, self.ybr, self.ybi) for bus in self.bus_idx[self.m[self.bus_q_idx]]])
        p_line_from = a([p_ij(i, j, v, delta, self.yfr, self.yfi) for i, j in zip(self.fbus[self.m[self.line_p_from_idx]], self.tbus[self.m[self.line_p_from_idx]])])
        q_line_from = a([q_ij(i, j, v, delta, self.yfr, self.yfi) for i, j in zip(self.fbus[self.m[self.line_q_from_idx]], self.tbus[self.m[self.line_q_from_idx]])])
        p_line_to = a([p_ji(j, i, v, delta, self.ytr, self.yti) for j, i in zip(self.tbus[self.m[self.line_p_to_idx]], self.fbus[self.m[self.line_p_to_idx]])])
        q_line_to = a([q_ji(j, i, v, delta, self.ytr, self.yti) for j, i in zip(self.tbus[self.m[self.line_q_to_idx]], self.fbus[self.m[self.line_q_to_idx]])])
        v_bus = v[self.bus_idx[self.m[self.bus_v_idx]]]
        hx = np.concatenate([p_bus, p_line_from, p_line_to, q_bus, q_line_from, q_line_to, v_bus])
        # print((a(self.eppci.z) - hx).sum())
        return hx

    def init_vm_from_measurements(self):
        vm_discrete = self.eppci.v
        vm_measured_mask = self.eppci.non_nan_meas_mask[self.bus_v_idx]
        v_measurements = self.eppci.z[-vm_measured_mask.sum():]
        vm_discrete[vm_measured_mask] = v_measurements
        vm_discrete[~vm_measured_mask] = v_measurements.mean()
        scales = 0.03 * onp.ones(len(vm_discrete))
        scales[vm_measured_mask] = self.eppci.r_cov[-vm_measured_mask.sum():]
        return vm_discrete, scales.clip(0., 0.03)

    
def se_model(hx: Hx, vm_discrete, vm_scales, va_non_slack_discrete, measurements, std_dev):
    # vm_dist = dist.Uniform(0.9 * np.ones(vm_discrete.shape[0]), 1.2 * np.ones(vm_discrete.shape[0]))
    vm_dist = dist.Normal(vm_discrete, vm_scales)
    vm_dist.arg_constraints = {
        "loc": interval(0.9, 1.1),
        "scale": greater_than(0)
    }
    vm = numpyro.sample("vm", vm_dist)
    # va_dist = dist.Uniform(va_non_slack_discrete - 0.05, va_non_slack_discrete + 0.05)
    va_dist = dist.Normal(va_non_slack_discrete, 0.05 * np.ones(va_non_slack_discrete.shape[0]))
    va_dist.arg_constraints = {
        "scale": greater_than(0)
    }
    va_dist = numpyro.sample("va", va_dist)
    va = np.concatenate([hx.vi_slack, va_dist])
    numpyro.sample("measurements", dist.Normal(hx(vm, va), std_dev), obs=measurements)


def se_guide(hx: Hx, vm_discrete, vm_scales, va_non_slack_discrete, measurements, std_dev):
    vm_mean = numpyro.param("vm_mean", vm_discrete)
    vm_scale = numpyro.param("vm_scale", vm_scales)
    vm_dist = dist.Normal(vm_mean, vm_scale)
    vm_dist.arg_constraints = {
        "loc": interval(0.9, 1.1),
        "scale": greater_than(0)
    }
    vm = numpyro.sample("vm", vm_dist)
    va_mean = numpyro.param("va_mean", va_non_slack_discrete)
    va_scale = numpyro.param("va_scale", 0.05 * np.ones(va_non_slack_discrete.shape[0]))
    va_dist = dist.Normal(va_mean, va_scale)
    va_dist.arg_constraints = {
        "scale": greater_than(0)
    }
    va_dist = numpyro.sample("va", va_dist)
    va = np.concatenate([hx.vi_slack, va_dist])
    hx_mean = numpyro.param("hx_mean", hx(vm, va))
    measurements = dist.Normal(hx_mean, std_dev)


class BayesAlgorithm(WLSAlgorithm):
    def estimate(self, eppci, opt_vars=None):
        warmup_samples, num_samples = 100, 900
        self.initialize(eppci)
        hx = Hx(eppci)
        vm_discrete, vm_scales = hx.init_vm_from_measurements()
        print(f"Initial V: {vm_discrete}")
        va_non_slack_discrete = eppci.delta[eppci.non_slack_buses]
        rng_key = random.PRNGKey(0)

        kernel = NUTS(se_model)
        mcmc = MCMC(kernel, warmup_samples, num_samples, num_chains=1, progress_bar=True)
        mcmc.run(rng_key, hx, a(vm_discrete), a(vm_scales), a(va_non_slack_discrete), a(eppci.z), a(eppci.r_cov))
        mcmc.print_summary()
        samples = mcmc.get_samples()

        # from numpyro.infer import SVI, ELBO
        # from numpyro.optim import Adam
        # from jax import lax
        # from numpyro.infer import Predictive
        # adam = Adam(0.1)
        # svi = SVI(se_model, se_guide, adam, ELBO())
        # rng_key, rng_key_init, rng_key_test = random.split(rng_key, 3)
        # vm_discrete, vm_scales, va_non_slack_discrete, z, r = a(vm_discrete), a(vm_scales), a(va_non_slack_discrete), a(eppci.z), a(eppci.r_cov)
        # svi_state = svi.init(rng_key_init, hx, vm_discrete, vm_scales, va_non_slack_discrete, z, r)
        #
        # @jit
        # def epoch_train(svi_state):
        #     def body_fn(i, val):
        #         loss_sum, svi_state = val
        #         svi_state, loss = svi.update(svi_state, hx, vm_discrete, vm_scales, va_non_slack_discrete, z, r)
        #         loss_sum += loss
        #         return loss_sum, svi_state
        #     return lax.fori_loop(0, 1, body_fn, (0., svi_state))
        #
        # for i in range(1000):
        #     loss, svi_state = epoch_train(svi_state)
        #     print(loss)
        #
        # predictive = Predictive(se_model, guide=se_guide, num_samples=num_samples, return_sites=("measurements", "vm", "va"))
        # samples = predictive(rng_key_test, hx, vm_discrete, vm_scales, va_non_slack_discrete, z, r)
        # print(samples["vm"][:, 1])

        eppci.update_E(onp.array(np.concatenate((samples["va"].mean(axis=0), samples["vm"].mean(axis=0)))))
        self.successful = True
        from bokeh.plotting import output_file
        import arviz as az
        output_file("bse1.html")
        az.style.use("arviz-darkgrid")
        p1 = az.plot_kde(samples["vm"][:, 1], samples["vm"][:, 2], backend="bokeh")
        output_file("bse2.html")
        p2 = az.plot_kde(samples["vm"][:, 0], quantiles=[0.05, 0.5, 0.95], backend="bokeh")
        output_file("bse3.html")
        p2 = az.plot_kde(samples["vm"][:, 2], quantiles=[0.05, 0.5, 0.95], backend="bokeh")
        return eppci


if __name__ == "__main__":
    from pandapower.test.estimation.test_wls_estimation import *

    # test_init_slack_with_multiple_transformers()
    # test_2bus()
    test_3bus_with_transformer()
    # test_3bus()
    # test_3bus_with_i_line_measurements()
    # test_cigre_network()
