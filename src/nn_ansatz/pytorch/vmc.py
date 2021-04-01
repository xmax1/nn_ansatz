
import torch as tc
import numpy as np
from torch import nn
from typing import Tuple

def clip(x_in):
    median = x_in.median()
    total_var = (x_in-median).abs().mean()
    clip_min = median - 5*total_var
    clip_max = median + 5*total_var
    x_out = tc.clamp(x_in, min=float(clip_min), max=float(clip_max))
    return x_out


def center(e_loc):
    e_loc_clipped = clip(e_loc)
    n_samples = len(e_loc)
    e_loc_centered = e_loc_clipped - e_loc_clipped.mean()
    return e_loc_centered


def get_energy_and_center(r_atoms, r_electrons, z_atoms, model):
    e_locs = compute_local_energy(r_atoms, r_electrons, z_atoms, model)
    e_mean = e_locs.mean()
    e_std = e_locs.std()
    e_locs_centered = center(e_locs)
    return e_locs_centered, e_mean


def batched_cdist_l2(x1, x2):
    """
    Notes:
        - the unsqueezes must be this way round to maintain the correct broadcasting for the a-e distances
    """
    x1_sq = (x1 ** 2).sum(-1, keepdims=True)
    x2_sq = (x2 ** 2).sum(-1, keepdims=True)
    cdist = (x1_sq.transpose(-1, -2) + x2_sq - (2 * x1.unsqueeze(1) * x2.unsqueeze(2)).sum(-1)).sqrt()
    return cdist

# def compute_ae_distances(x1, x2):
#     x1_sq = (x1 ** 2).sum(-1, keepdims=True)
#     x2_sq = (x2 ** 2).sum(-1, keepdims=True)
#     cdist = (x2_sq.transpose(-1, -2) + x1_sq - (2 * x1.unsqueeze(1) * x2.unsqueeze(2)).sum(-1)).sqrt()
#     return cdist


def compute_local_energy(wf, walkers, r_atoms, z_atoms):

    first_order_squared, second_order = laplacian(wf, walkers)
    potential_energy = compute_potential_energy(walkers, r_atoms, z_atoms)

    return -0.5 * (second_order.sum(-1) + first_order_squared.sum(-1)) + potential_energy


def laplacian(model: nn.Module, walkers: tc.Tensor) -> Tuple[tc.Tensor, tc.Tensor]:
    n_walkers, n_el = walkers.shape[:2]
    n_r = 3 * n_el

    walkers = walkers.reshape(n_walkers, n_r)
    r_s = [walkers[..., i].requires_grad_(True) for i in range(n_r)]
    positions = tc.stack(r_s, -1)
    walkers = positions.view(n_walkers, n_el, 3)
    log_phi = model(walkers)

    dlogphi_dr = tc.autograd.grad(log_phi.sum(), positions, create_graph=True, retain_graph=True)[0]
    grads = tc.split(dlogphi_dr, 1, -1)

    d2logphi_dr2 = tc.stack([tc.autograd.grad(grad.sum(), r, retain_graph=True)[0] for grad, r in zip(grads, r_s)], 1)

    return dlogphi_dr**2, d2logphi_dr2


def compute_potential_energy(walkers, r_atoms, z_atom):
    # walkers (m, n_el, 3)
    # r_atoms (1, n_atoms, 3)
    # z_atoms (n_atoms)

    e_e_dist = batched_cdist_l2(walkers, walkers)  # electron - electron distances
    potential_energy = tc.tril(1. / e_e_dist, diagonal=-1).sum((-1, -2))

    a_e_dist = batched_cdist_l2(walkers, r_atoms)  # atom - electron distances
    z_atom = z_atom.unsqueeze(0).unsqueeze(2)
    potential_energy -= (z_atom / a_e_dist).sum((-1, -2))

    # if n_atom > 1:
    #     a_a_dist = batched_cdist_l2(r_atoms, r_atoms)
    #     weighted_a_a = tc.einsum('bn,bm,bnm->bnm', z_atom, z_atom, 1/a_a_dist)
    #     unique_a_a = weighted_a_a[:, np.tril(np.ones((n_atom, n_atom), dtype=bool), -1)]  # this will not work
    #     potential_energy += unique_a_a.sum(-1)

    return potential_energy
