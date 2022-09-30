import torch as tc
from torch import nn
import numpy as np
from torch.autograd import Variable, Function
from typing import Tuple
import inspect

INIT = 0.01


def logabssumdet(up_orb, down_orb):
    sa, loga = tc.slogdet(up_orb)
    sb, logb = tc.slogdet(down_orb)
    log_sum = loga + logb
    log_sum_max = tc.max(log_sum, dim=1, keepdim=True)[0]
    log_sum_center = log_sum - log_sum_max
    logpsi = tc.log(tc.abs(tc.sum(sa * sb * tc.exp(log_sum_center), dim=1))) + tc.squeeze(log_sum_max)
    return logpsi


class LinearSingle(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device,
                 dtype):
        super(LinearSingle, self).__init__()

        w = tc.normal(0., INIT, (in_dim + 1, out_dim)).to(device=device, dtype=dtype)
        self.w = nn.Parameter(w)

    def forward(self, data: tc.Tensor, split_preactivations: tc.Tensor, residual: tc.Tensor) -> tc.Tensor:
        bias_data = tc.ones((*data.shape[:2], 1), device=data.device, dtype=data.dtype)

        data_w_bias = tc.cat((data, bias_data), dim=-1)
        argument = data_w_bias @ self.w + split_preactivations
        outputs = argument.tanh() + residual
        return outputs


class LinearPairwise(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device,
                 dtype):
        super(LinearPairwise, self).__init__()

        w = tc.normal(0., INIT, (in_dim + 1, out_dim)).to(device=device, dtype=dtype)
        self.w = nn.Parameter(w)

    def forward(self, data: tc.Tensor, residual: tc.Tensor) -> tc.Tensor:
        bias_data = tc.ones((*data.shape[:2], 1), device=data.device, dtype=data.dtype)

        data_w_bias = tc.cat((data, bias_data), dim=-1)
        argument = data_w_bias @ self.w
        outputs = argument.tanh() + residual
        return outputs


class LinearSplit(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device,
                 dtype):
        super(LinearSplit, self).__init__()

        w = tc.normal(0., INIT, (in_dim, out_dim)).to(device=device, dtype=dtype)
        self.w = nn.Parameter(w)

    def forward(self, data: tc.Tensor) -> tc.Tensor:
        out = data @ self.w
        return out


class Mixer():
    def __init__(self,
                 n_single_features,
                 n_pairwise_features,
                 n_electrons,
                 n_spin_up,
                 diagonal,
                 device,
                 dtype):
        super(Mixer, self).__init__()

        n_spin_down = n_electrons - n_spin_up
        n_pairwise = n_electrons ** 2

        self.n_electrons = n_electrons
        self.n_spin_up = float(n_spin_up)
        self.n_spin_down = float(n_spin_down)

        tmp1 = tc.ones((1, n_spin_up, n_single_features), dtype=tc.bool)
        tmp2 = tc.zeros((1, n_spin_down, n_single_features), dtype=tc.bool)
        self.spin_up_mask = tc.cat((tmp1, tmp2), dim=1).to(device=device, dtype=dtype)
        self.spin_down_mask = (~tc.cat((tmp1, tmp2), dim=1)).to(device=device, dtype=dtype)

        self.pairwise_spin_up_mask, self.pairwise_spin_down_mask = \
                generate_pairwise_masks(n_electrons, n_pairwise, n_spin_up, n_pairwise_features, diagonal)
        self.pairwise_spin_up_mask, self.pairwise_spin_down_mask = \
            self.pairwise_spin_up_mask.to(device=device, dtype=dtype), self.pairwise_spin_down_mask.to(device=device, dtype=dtype)

    def __call__(self, single: tc.Tensor, pairwise: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
        # single (n_samples, n_electrons, n_single_features)
        # pairwise (n_samples, n_electrons, n_pairwise_features)
        # spin_up_mask = self.spin_up_mask.repeat((n_samples, 1, 1))
        # spin_down_mask = self.spin_down_mask.repeat((n_samples, 1, 1))

        # --- Single summations
        # up
        sum_spin_up = self.spin_up_mask.type(single.dtype) * single
        sum_spin_up = sum_spin_up.sum(1, keepdim=True) / self.n_spin_up
        # sum_spin_up = sum_spin_up.repeat((1, self.n_electrons, 1))

        # down
        sum_spin_down = self.spin_down_mask.type(single.dtype) * single
        sum_spin_down = sum_spin_down.sum(1, keepdim=True) / self.n_spin_down
        # sum_spin_down = sum_spin_down.repeat((1, self.n_electrons, 1))

        # --- Pairwise summations
        sum_pairwise = pairwise.unsqueeze(1).repeat((1, self.n_electrons, 1, 1))

        # up
        sum_pairwise_up = self.pairwise_spin_up_mask.type(single.dtype) * sum_pairwise
        sum_pairwise_up = sum_pairwise_up.sum(2) / self.n_spin_up

        # down
        sum_pairwise_down = self.pairwise_spin_down_mask.type(single.dtype) * sum_pairwise
        sum_pairwise_down = sum_pairwise_down.sum(2) / self.n_spin_down

        features = tc.cat((single, sum_pairwise_up, sum_pairwise_down), dim=2)
        split_features = tc.cat((sum_spin_up, sum_spin_down), dim=2)
        return features, split_features


def compute_inputs(r_electrons, n_samples : int, ae_vectors, n_atoms : int, n_electrons : int)\
        -> Tuple[tc.Tensor, tc.Tensor]:
    # r_atoms: (n_atoms, 3)
    # r_electrons: (n_samples, n_electrons, 3)
    # ae_vectors: (n_samples, n_electrons, n_atoms, 3)
    ae_distances = tc.norm(ae_vectors, dim=-1, keepdim=True)
    single_inputs = tc.cat((ae_vectors, ae_distances), dim=-1)
    single_inputs = single_inputs.view((-1, n_electrons, 4 * n_atoms))

    re1 = r_electrons.unsqueeze(2)
    re2 = re1.permute((0, 2, 1, 3))
    ee_vectors = re1 - re2

    mask = tc.eye(n_electrons, dtype=tc.bool)
    mask = ~mask.unsqueeze(0).unsqueeze(3).repeat((n_samples, 1, 1, 3))

    ee_vectors = ee_vectors[mask]
    ee_vectors = ee_vectors.view((-1, int(n_electrons ** 2 - n_electrons), 3))
    ee_distances = tc.norm(ee_vectors, dim=-1, keepdim=True)

    pairwise_inputs = tc.cat((ee_vectors, ee_distances), dim=-1)

    return single_inputs, pairwise_inputs


def compute_ae_vectors(r_atoms: tc.Tensor, r_electrons: tc.Tensor) -> tc.Tensor:
    # ae_vectors (n_samples, n_electrons, n_atoms, 3)
    r_atoms = r_atoms.unsqueeze(1)
    r_electrons = r_electrons.unsqueeze(2)
    ae_vectors = r_electrons - r_atoms
    return ae_vectors


def generate_pairwise_masks(n_el: int, n_pairwise: int, n_spin_up: int, n_pairwise_features: int, diagonal) \
        -> Tuple[tc.Tensor, tc.Tensor]:
    n_pairwise = int(n_el**2 - int(not diagonal) * n_el)

    eye_mask = ~np.eye(n_el, dtype=bool)
    ups = np.ones(n_el, dtype=bool)
    ups[n_spin_up:] = False
    downs = ~ups

    spin_up_mask = []
    spin_down_mask = []
    mask = np.zeros((n_el, n_el), dtype=bool)

    for electron in range(n_el):
        e_mask_up = np.zeros((n_el,), dtype=bool)
        e_mask_down = np.zeros((n_el,), dtype=bool)

        mask_up = np.copy(mask)
        mask_up[electron, :] = ups
        mask_up = mask_up[eye_mask].reshape(-1)

        mask_down = np.copy(mask)
        mask_down[electron, :] = downs
        mask_down = mask_down[eye_mask].reshape(-1)

        if diagonal:
            if electron < n_spin_up:
                e_mask_up[electron] = True
            mask_up = np.concatenate((mask_up, e_mask_up), axis=0)

            if electron >= n_spin_up:
                e_mask_down[electron] = True
            mask_down = np.concatenate((mask_down, e_mask_down), axis=0)

        spin_up_mask.append(mask_up)
        spin_down_mask.append(mask_down)


    spin_up_mask = tc.tensor(spin_up_mask, dtype=tc.bool)
    # (n_samples, n_el, n_el, n_pairwise_features)
    spin_up_mask = spin_up_mask.view((1, n_el, n_pairwise, 1))
    spin_up_mask = spin_up_mask.repeat((1, 1, 1, n_pairwise_features))

    spin_down_mask = tc.tensor(spin_down_mask, dtype=tc.bool)
    spin_down_mask = spin_down_mask.view((1, n_el, n_pairwise, 1))
    spin_down_mask = spin_down_mask.repeat((1, 1, 1, n_pairwise_features))

    return spin_up_mask, spin_down_mask


class EnvelopeLinear(nn.Module):
    def __init__(self,
                 n_hidden,
                 n_spin_det,
                 n_determinants,
                 device,
                 dtype):
        super(EnvelopeLinear, self).__init__()

        w = tc.normal(0., INIT, (n_determinants, n_spin_det, n_hidden + 1)).to(device=device, dtype=dtype)
        self.w = nn.Parameter(w, requires_grad=True)

    def forward(self, data):
        n_samples, n_spin_det = data.shape[:2]

        bias_data = tc.ones((n_samples, n_spin_det, 1), device=data.device, dtype=data.dtype)

        data_w_bias = tc.cat((data, bias_data), dim=-1)
        out = tc.einsum('njf,kif->njki', data_w_bias, self.w)
        return out


class EnvelopeSigma(nn.Module):
    def __init__(self,
                 n_spin_det,
                 n_determinants,
                 n_atoms,
                 device,
                 dtype):
        super(EnvelopeSigma, self).__init__()

        sigma_shape = (n_determinants, n_spin_det, n_atoms, 3, 3)
        sigma = tc.eye(3).view(1, 1, 1, 3, 3).repeat(n_determinants, n_spin_det, n_atoms, 1, 1) + tc.normal(0., INIT, sigma_shape)
        sigma = sigma.to(device=device, dtype=dtype)
        self.sigma_einsum = nn.Parameter(sigma, requires_grad=True)

    def forward(self, ae_vectors):
        pa_einsum = tc.einsum('njmv,kimvc->njkimc', ae_vectors, self.sigma_einsum)
        exponential = tc.exp(-tc.norm(pa_einsum, dim=-1))
        return exponential


class EnvelopePi(nn.Module):
    def __init__(self,
                 n_spin_det,
                 n_determinants,
                 n_atoms,
                 device,
                 dtype):
        super(EnvelopePi, self).__init__()

        pi_shape = (n_determinants, n_spin_det, n_atoms)
        pi = tc.normal(0., INIT, pi_shape)
        pi = tc.ones(pi_shape).to(device=device, dtype=dtype)
        self.pi = nn.Parameter(pi, requires_grad=True)

    def forward(self, factor, exponential):
        exp = tc.einsum('njkim,kim->njki', exponential, self.pi).contiguous()
        output = factor * exp
        output = output.permute((0, 2, 1, 3))
        return output.contiguous()


class fermiNet(nn.Module):
    def __init__(self,
                 mol,
                 diagonal: bool = False):
        super(fermiNet, self).__init__()
        self.device = mol.device
        self.dtype = mol.dtype
        dv, dt = self.device, self.dtype
        self.diagonal = diagonal

        n_layers, n_sh, n_ph, n_det = mol.n_layers, mol.n_sh, mol.n_ph, mol.n_det
        r_atoms, n_el, n_up, n_atoms = mol.r_atoms, mol.n_el, mol.n_up, mol.n_atoms
        #r_atoms = from_np(r_atoms)

        # things we need
        self.n_layers = mol.n_layers
        self.r_atoms = r_atoms
        self.n_el = int(n_el)
        self.n_pairwise = int(n_el ** 2 - int(not diagonal) * n_el)
        self.n_up = n_up
        self.n_down = n_el - n_up
        self.n_atoms = int(n_atoms)
        n_down = n_el - n_up
        self.n_determinants = mol.n_det

        # layers
        s_in = 4 * n_atoms
        p_in = 4
        s_hidden = mol.n_sh
        self.s_hidden = s_hidden
        p_hidden = mol.n_ph
        self.p_hidden = p_hidden
        s_mixed_in = 4 * n_atoms + 4 * 2
        s_mixed = mol.n_sh * 3 + mol.n_ph * 2

        self.mix_in = Mixer(s_in, p_in, n_el, n_up, diagonal, dv, dt)
        self.lin_split_in = LinearSplit(2 * s_in, s_hidden, dv, dt)

        self.stream_s0 = LinearSingle(s_mixed_in, s_hidden, dv, dt)
        self.stream_p0 = LinearPairwise(p_in, p_hidden, dv, dt)
        self.m0 = Mixer(s_hidden, p_hidden, n_el, n_up, diagonal, dv, dt)

        self.single_splits = \
            tc.nn.ModuleList([LinearSplit(2 * s_hidden, s_hidden, dv, dt) for _ in range(n_layers)])
        self.single_intermediate = \
            tc.nn.ModuleList([LinearSingle(s_mixed - 2 * s_hidden, s_hidden, dv, dt) for _ in range(n_layers)])
        self.pairwise_intermediate = \
            tc.nn.ModuleList([LinearPairwise(p_hidden, p_hidden, dv, dt) for _ in range(n_layers)])
        self.intermediate_mix = Mixer(s_hidden, p_hidden, n_el, n_up, diagonal, dv, dt)

        self.env_up_linear = EnvelopeLinear(s_hidden, n_up, n_det, dv, dt)
        self.env_up_sigma = EnvelopeSigma(n_up, n_det, n_atoms, dv, dt)
        self.env_up_pi = EnvelopePi(n_up, n_det, n_atoms, dv, dt)

        self.env_down_linear = EnvelopeLinear(s_hidden, n_down, n_det, dv, dt)
        self.env_down_sigma = EnvelopeSigma(n_down, n_det, n_atoms, dv, dt)
        self.env_down_pi = EnvelopePi(n_down, n_det, n_atoms, dv, dt)

        print('Model: \n',
              'device   = %s \n' % self.device,
              'n_sh     = %i \n' % n_sh,
              'n_ph     = %i \n' % n_ph,
              'n_layers = %i \n' % n_layers,
              'n_det    = %i \n' % n_det)

    def layers(self):
        for m in self.children():
            if len(list(m.parameters())) == 0:
                continue
            elif isinstance(m, tc.nn.ModuleList):
                yield from m
            else:
                yield m

    def forward(self, walkers):

        up_orbitals, down_orbitals = self.generate_orbitals(walkers)

        # logabssumdet
        log_psi = logabssumdet(up_orbitals, down_orbitals)

        return log_psi

    def generate_orbitals(self, walkers):
        #walkers = from_np(walkers)
        n_walkers = int(walkers.shape[0])

        self.single_input_residual = tc.zeros((n_walkers, self.n_el, self.s_hidden), device=walkers.device, dtype=walkers.dtype)
        self.pairwise_input_residual = tc.zeros((n_walkers, self.n_pairwise, self.p_hidden), device=walkers.device, dtype=walkers.dtype)

        ae_vectors = compute_ae_vectors(self.r_atoms, walkers)

        # the inputs
        single, pairwise = compute_inputs(walkers, n_walkers, ae_vectors, self.n_atoms, self.n_el)

        if self.diagonal:
            diagonal_pairwise_input = tc.zeros((n_walkers, self.n_el, 4), device=walkers.device, dtype=walkers.dtype)
            pairwise = tc.cat((pairwise, diagonal_pairwise_input), dim=1)

        # mix in
        single_mixed, single_split = self.mix_in(single, pairwise)

        # first layer
        single_split = self.lin_split_in(single_split)
        single = self.stream_s0(single_mixed, single_split, self.single_input_residual)
        pairwise = self.stream_p0(pairwise, self.pairwise_input_residual)

        # intermediate layers
        for ss, ls, ps in zip(self.single_intermediate, self.single_splits, self.pairwise_intermediate):
            single_mixed, single_split = self.intermediate_mix(single, pairwise)

            single_split = ls(single_split)
            single = ss(single_mixed, single_split, single)
            pairwise = ps(pairwise, pairwise)

        # single_mixed = tc.cat((single_mixed, single_split.repeat(1, self.n_el, 1)), dim=2)
        # envelopes
        ae_vectors_up, ae_vectors_down = ae_vectors.split([self.n_up, self.n_down], dim=1)
        data_up, data_down = single.split([self.n_up, self.n_down], dim=1)

        factor_up = self.env_up_linear(data_up)
        factor_down = self.env_down_linear(data_down)

        exponent_up = self.env_up_sigma(ae_vectors_up)
        exponent_down = self.env_down_sigma(ae_vectors_down)

        up_orbitals = self.env_up_pi(factor_up, exponent_up)
        down_orbitals = self.env_down_pi(factor_down, exponent_down)

        return up_orbitals, down_orbitals
