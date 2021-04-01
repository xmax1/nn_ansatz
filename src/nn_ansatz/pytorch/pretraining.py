
import torch as tc
from torch.distributions import Normal
import numpy as np
from torch import nn
from tqdm.auto import trange
import torch
from .vmc import compute_local_energy
from .sampling import MetropolisHasting

# mol = gto.Mole()
# mol.atom = """
# Ne 0.0 0.0 0.0
# """
# # mol.charge = 1
# # mol.spin = 1  # is the number of unpaired electrons
#
# mol.unit = "Bohr"
# mol.basis = "sto3g"
# # mol.verbose = 4
# mol.build()
#
# m = scf.RHF(mol)
# print('E(HF) = %g' % m.kernel())
#
# number_of_atoms = mol.natm
# conv, e, mo_e, mo, mo_occ = scf.rhf.kernel(scf.hf.SCF(mol), dm0=np.eye(mol.nao_nr()))


# def reader(path):
#     mol = gto.Mole()
#     with open(path, 'rb') as f:
#         data = pk.load(f)
#     mol.atom = data["mol"]
#     mol.unit = "Bohr"
#     mol.basis = data["basis"]
#     mol.verbose = 4
#     mol.spin = data["spin"]
#     # mol.charge = 1
#     mol.build()
#     number_of_electrons = mol.tot_electrons()
#     number_of_atoms = mol.natm
#     ST = data["super_twist"]
#     print('atom: ', mol.atom)
#     # mol
#     return ST, mol


class Pretrainer():
    def __init__(self,
                 mol,
                 n_pretrain_iterations: int = 1000):

        self.mol = mol

        self.n_up = mol.n_up
        self.n_down = mol.n_down
        self.n_el = mol.n_el

        self.n_iterations = n_pretrain_iterations

    def compute_orbital_probability(self, samples):
        up_dets, down_dets = self.hf_orbitals(samples)

        spin_ups = up_dets ** 2
        spin_downs = down_dets ** 2

        p_up = tc.diagonal(spin_ups, dim1=-2, dim2=-1).prod(-1)
        p_down = tc.diagonal(spin_downs, dim1=-2, dim2=-1).prod(-1)
        # p_up = spin_ups.prod(1).prod(1)
        # p_down = spin_downs.prod(1).prod(1)

        probabilities = p_up * p_down

        return probabilities

    def pyscf_call(self, samples):
        samples = samples.cpu().numpy()
        ao_values = self.mol.pyscf_mol.eval_gto("GTOval_cart", samples)
        return tc.from_numpy(ao_values)

    def hf_orbitals(self, coord):
        coord = coord.view((-1, 3))

        number_spin_down = self.n_down
        number_spin_up = self.n_el - number_spin_down

        ao_values = self.pyscf_call(coord).to(device=coord.device, dtype=coord.dtype)
        ao_values = ao_values.view((int(len(ao_values) / self.n_el), self.n_el, len(ao_values[0])))

        spin_up = tc.stack([(self.mol.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
             for orb_number in range(number_spin_up) for el_number in
             range(number_spin_up)], dim=1).view((-1, number_spin_up, number_spin_up))

        spin_down = tc.stack([(self.mol.moT[orb_number, :] * ao_values[:, el_number, :]).sum(-1)
                            for orb_number in range(number_spin_down) for el_number in
                            range(number_spin_up, self.n_el)], dim=1).view((-1, number_spin_down, number_spin_down))

        return spin_up, spin_down

    # def compute_grads(self, model, samples):
    #     model_up_dets, model_down_dets = model(samples)
    #     n_det = model_up_dets.shape[1]
    #
    #     up_dets, down_dets = self.wave_function(samples)
    #     up_dets = tile_labels(up_dets, n_det).to(model.device)
    #     down_dets = tile_labels(down_dets, n_det).to(model.device)
    #
    #     loss = mse_error(up_dets, model_up_dets)
    #     loss += mse_error(down_dets, model_down_dets)
    #     model.zero_grad()
    #     loss.backward()  # in order for hook to work must call backward
    #     grads = [w.grad.data for w in list(model.parameters())[:-1]]
    #     return grads

    def pretrain(self,
                 wf,
                 walkers,
                 n_it: int = 1000,
                 lr: float = 1e-4):
        device, dtype = wf.device, wf.dtype
        sampler = MetropolisHastingsPretrain()
        wf_walkers = walkers

        wf_sampler = MetropolisHasting(wf)
        for i in range(500):
            wf_walkers, wf_acc = wf_sampler(wf_walkers)
            e_locs = compute_local_energy(wf, wf_walkers, self.mol.r_atoms, self.mol.z_atoms)
            print(e_locs.mean())


        opt = tc.optim.Adam(list(wf.parameters()), lr=lr)
        steps = trange(
            0,  # init_step = 0
            n_it,
            initial=0,
            total=n_it,
            desc='pretraining',
            disable=None,
        )

        # walkers = initialize_walkers(self.mol.n_el_atoms, self.mol.atom_positions, n_walkers).to(device=device, dtype=dtype)

        for step in steps:
            wf_walkers, wf_acc = wf_sampler(wf_walkers)
            e_locs = compute_local_energy(wf, wf_walkers, self.mol.r_atoms,  self.mol.z_atoms)

            walkers = sampler(wf, self, walkers)

            up_dets, down_dets = self.hf_orbitals(walkers)
            up_dets = tile_labels(up_dets, wf.n_determinants)
            down_dets = tile_labels(down_dets, wf.n_determinants)

            model_up_dets, model_down_dets = wf.generate_orbitals(walkers)

            loss = mse_error(up_dets, model_up_dets)
            loss += mse_error(down_dets, model_down_dets)
            opt.zero_grad()
            loss.backward()  # in order for hook to work must call backward
            opt.step()

            print('step %i | e_mean %.4f  | loss %.2f ' % (step, e_locs.mean().detach().cpu().numpy(), loss) )

            # steps.set_postfix(E=f'{e_locs.mean().detach().cpu().numpy():.6f}')
            # steps.refresh()


def mse_error(targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    return ((targets - outputs)**2).sum((1, 2, 3)).mean(0)


def tile_labels(label: torch.Tensor, n_k: int) -> torch.Tensor:
    x = label.unsqueeze(dim=1).repeat((1, n_k, 1, 1))
    return x


class RandomWalker():
    r""" Creates normal sampler with std of sigma

    Used to suggest new updates to the positions of the walkers

    Usage:
        distr = RandomWalker(sigma.to(device=device, dtype=dtype))

    Args:
        sigma (float): step size of the walkers

    """
    def __init__(self, sigma):
        self.step_gaussian = Normal(0.0, sigma)

    def resample(self, prev_sample) -> torch.Tensor:
        return prev_sample + self.step_gaussian.sample(prev_sample.shape)


class Uniform(nn.Module):
    r""" Creates a uniform sampler between 0 and 1

    Used to determine whether moves accepted or rejected

    Usage:
        alpha_distr = Uniform(torch.tensor(0.).to(device=device, dtype=dtype), torch.tensor(1.).to(device=device, dtype=dtype))

    """
    def __init__(self, low=0, high=1):
        super(Uniform, self).__init__()
        self.low = torch.tensor(low) if type(low) != torch.Tensor else low
        self.high = torch.tensor(high) if type(high) != torch.Tensor else high

    def forward(self, batch_size: int = 1):
        return self.low + torch.rand(batch_size, device=self.low.device) * (self.high - self.low)

    def sample(self, batch_size: int = 1):
        return self(batch_size)


class ToProb(nn.Module):
    def forward(self, amps: torch.Tensor) -> torch.Tensor:
        return torch.exp(amps) ** 2


def initialize_walkers(ne_atoms, atom_positions, n_walkers):
    r""" Initialises walkers for pretraining

        Usage:
            walkers = initialize_walkers(ne_atoms, atom_positions, n_walkers).to(device=self.device, dtype=self.dtype)

        Args:
            ne_atoms (list int): number of electrons assigned to each nucleus
            atom_positions (list np.array): positions of the nuclei
            n_walkers (int): number of walkers

        Returns:
            walkers (np.array): walker positions (n_walkers, n_elec, 3)

        """
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = np.random.normal(loc=atom_position, scale=1., size=(n_walkers, 1, 3))
                ups.append(curr_sample_up)
            else:
                curr_sample_down = np.random.normal(loc=atom_position, scale=1., size=(n_walkers, 1, 3))
                downs.append(curr_sample_down)
    ups = np.concatenate(ups, axis=1)
    downs = np.concatenate(downs, axis=1)
    curr_sample = np.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    return torch.from_numpy(curr_sample)


class MetropolisHastingsPretrain(nn.Module):
    r""" Implements MetropolisHastings sampling based on [pfau2020ab]

    walkers congigurations based on the amplitudes of both the Hartree Fock orbitals and the wave function Ansatz

    .. math:

    Usage:
        sampler = MetropolisHastingsPretrain()

    Args:
        sigma (float): step size for the walkers (std of the proposed moves)
        correlation_length (int): number of steps between sampling each update of the walker positions
        target_acceptance (float): the target acceptance of the steps

    Returns:
        curr_sample (torch.Tensor): walker configurations (n_walkers, n_elec, 3)

    """
    def __init__(self,
                 sigma: float = 0.02,
                 correlation_length: int = 10,
                 target_acceptance: float = 0.5):
        super(MetropolisHastingsPretrain, self).__init__()

        self.sigma = sigma
        self.correlation_length = correlation_length

        self.distr = RandomWalker(sigma)
        self.alpha_distr = Uniform()
        self.to_prob = ToProb()

        self.acceptance = 0.0
        self.target_acceptance = target_acceptance

        print('initialized pretraining sampler')

    def forward(self, wf, pretrainer, curr_walkers):
        device, dtype = curr_walkers.device, curr_walkers.dtype
        n_walkers = curr_walkers.shape[0]

        # --- split the walkers and walkers half from the hf_orbitals and half from the wave function
        sams = curr_walkers.split([n_walkers // 2, n_walkers // 2])
        curr_walkers_model, curr_walkers_hf = sams[0].squeeze(), sams[1].squeeze()
        shape = curr_walkers_model.shape

        curr_log_amp = wf(curr_walkers_model)[0]
        curr_prob_model = self.to_prob(curr_log_amp)
        curr_prob_hf = pretrainer.compute_orbital_probability(curr_walkers_hf)

        acceptance_total_mod = 0.
        acceptance_total_hf = 0.
        for _ in range(self.correlation_length):
            # --- next walkers
            new_walkers_model = curr_walkers_model + tc.normal(0.0, self.sigma, size=shape, device=device, dtype=dtype)
            new_log_amp = wf(new_walkers_model)[0]
            new_prob_model = self.to_prob(new_log_amp)

            new_walkers_hf = curr_walkers_hf + tc.normal(0.0, self.sigma, size=shape, device=device)
            new_prob_hf = pretrainer.compute_orbital_probability(new_walkers_hf)

            # --- update walkers
            alpha_model = new_prob_model / curr_prob_model
            alpha_hf = new_prob_hf / curr_prob_hf

            # --- generate masks
            mask_model = alpha_model > tc.rand(shape[0], device=device, dtype=dtype)
            mask_hf = alpha_hf > tc.rand(shape[0], device=device, dtype=dtype)

            curr_walkers_model = tc.where(mask_model.unsqueeze(-1).unsqueeze(-1), new_walkers_model, curr_walkers_model)
            curr_prob_model = tc.where(mask_model, new_prob_model, curr_prob_model)

            curr_walkers_hf = tc.where(mask_hf.unsqueeze(-1).unsqueeze(-1), new_walkers_hf, curr_walkers_hf)
            curr_prob_hf = tc.where(mask_hf, new_prob_hf, curr_prob_hf)

            acceptance_total_mod += mask_model.type(dtype).mean()
            acceptance_total_hf += mask_hf.type(dtype).mean()

        curr_walkers = tc.cat([curr_walkers_model, curr_walkers_hf], dim=0)
        # --- randomly permute so some walkers in the next run walkers from different distribution than in this run
        idxs = tc.randperm(len(curr_walkers))
        curr_walkers = curr_walkers[idxs]
        return curr_walkers

    def adjust_sampling_steps(self, acceptance):
        if acceptance < 0.5:
            self.sigma += 0.001
        else:
            self.sigma -= 0.001


