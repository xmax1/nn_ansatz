import jax.random as rnd
import jax.numpy as jnp
from jax import vmap, jit, grad
from jax.experimental.optimizers import adam
from jax import tree_util
from tqdm.notebook import trange


from .sampling import create_sampler
from .ansatz import create_wf
from .parameters import initialise_params, initialise_d0s, expand_d0s
from .systems import SystemAnsatz
from .pretraining import pretrain_wf
from .vmc import create_energy_fn, create_grad_function
# from .utils import *
from .optimisers import create_natural_gradients_fn, kfac
from .utils import Logging, load_pk, save_pk




def validate_ewalds(r_atoms=None,
                    z_atoms=None,
                    n_el=None,
                    n_el_atoms=None,
                    periodic_boundaries=False,
                    cell_basis=None,
                    unit_cell_length=None,
        
                    opt: str = 'kfac',
                    lr: float = 1e-4,
                    damping: float = 1e-4,
                    norm_constraint: float = 1e-4,
                    n_it: int = 1000,
                    n_walkers: int = 1024,
                    step_size: float = 0.02,
        
                    n_layers: int = 2,
                    n_sh: int = 64,
                    n_ph: int = 16,
                    n_det: int = 2,
        
                    pre_lr: float = 1e-4,
                    n_pre_it: int = 1000,
                    load_pretrain: bool = False,
                    pretrain: bool = False,
                    pre_path: str = '',
        
                    seed: int = 369,
                    **kwargs):

    # load in a model

    # equilibrate

    # set kappa, real_cut, reciprocal_cut

    # compute the potential of those walkers
    logger = Logging(**kwargs)

    key = rnd.PRNGKey(seed)

    mol = SystemAnsatz(r_atoms,
                       z_atoms,
                       n_el,
                       unit_cell_length=unit_cell_length,
                       cell_basis=cell_basis,
                       periodic_boundaries=periodic_boundaries,
                       n_el_atoms=n_el_atoms,
                       n_layers=n_layers,
                       n_sh=n_sh,
                       n_ph=n_ph,
                       n_det=n_det,
                       step_size=step_size,
                       **kwargs)

    wf, kfac_wf, wf_orbitals = create_wf(mol)
    params = initialise_params(key, mol)
    d0s = expand_d0s(initialise_d0s(mol), n_walkers)
    sampler, equilibrate = create_sampler(wf, mol, correlation_length=10)
    walkers = mol.initialise_walkers(n_walkers=n_walkers)
    params, walkers = load_pk(pre_path)
    walkers = equilibrate(params, walkers, d0s, key, n_it=1000, step_size=0.02 ** 2)

    real_basis = mol.real_basis
    reciprocal_basis = mol.reciprocal_basis
    kappa = mol.kappa
    mesh = [mol.reciprocal_cut for i in range(3)]
    volume = mol.volume

    real_lattice = generate_real_lattice(real_basis, mol.real_cut, mol.reciprocal_height)  # (n_lattice, 3)
    reciprocal_lattice = generate_reciprocal_lattice(reciprocal_basis, mesh)
    rl_inner_product = inner(reciprocal_lattice, reciprocal_lattice)
    rl_factor = (4*jnp.pi / volume) * jnp.exp(- rl_inner_product / (4*kappa**2)) / rl_inner_product  

    e_charges = jnp.array([-1. for i in range(mol.n_el)])
    charges = jnp.concatenate([mol.z_atoms, e_charges], axis=0)  # (n_particle, )
    q_q = charges[None, :] * charges[:, None]  # q_i * q_j  (n_particle, n_particle)

    def compute_potential_energy_solid_i(walkers, r_atoms, z_atoms):

        """
        :param walkers (n_el, 3):
        :param r_atoms (n_atoms, 3):
        :param z_atoms (n_atoms, ):

        Pseudocode:
            - compute the potential energy (pe) of the cell
            - compute the pe of the cell electrons with electrons outside
            - compute the pe of the cell electrons with nuclei outside
            - compute the pe of the cell nuclei with nuclei outside
        """

        # put the walkers and r_atoms together
        walkers = jnp.concatenate([r_atoms, walkers], axis=0)  # (n_particle, 3)

        # compute the Rs0 term
        p_p_vectors = vector_sub(walkers, walkers)
        p_p_distances = compute_distances(walkers, walkers)
        # p_p_distances[p_p_distances < 1e-16] = 1e200  # doesn't matter, diagonal dropped, this is just here to suppress the error
        Rs0 = jnp.tril(erfc(kappa * p_p_distances) / p_p_distances, k=-1)  # is half the value

        # compute the Rs > 0 term
        ex_walkers = vector_add(walkers, real_lattice)  # (n_particle, n_lattice, 3)
        tmp = walkers[:, None, None, :] - ex_walkers[None, ...]  # (n_particle, n_particle, n_lattice, 3)
        ex_distances = jnp.sqrt(jnp.sum(tmp**2, axis=-1))  
        Rs1 = jnp.sum(erfc(kappa * ex_distances) / ex_distances, axis=-1)
        real_sum = 0.5 * (q_q * (Rs0 + Rs1)).sum((-1, -2))
        
        # compute the constant factor
        self_interaction = - 0.5 * jnp.diag(q_q * 2 * kappa / jnp.sqrt(jnp.pi)).sum()
        constant = - 0.5 * charges.sum()**2 * jnp.pi / (kappa**2 * volume)  # is zero in neutral case

        # compute the reciprocal term reuse the ee vectors
        exp = jnp.real(jnp.sum(rl_factor[None, None, :] * jnp.exp(1j * p_p_vectors @ jnp.transpose(reciprocal_lattice)), axis=-1))
        reciprocal_sum = 0.5 * (q_q * exp).sum((-1,-2))
        
        potential = real_sum + reciprocal_sum + constant + self_interaction
        return real_sum, reciprocal_sum, potential

    compute_potential_energy = vmap(compute_potential_energy_solid_i, in_axes=(0, None, None))




