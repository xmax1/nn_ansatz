
import torch as tc
from torch.distributions import Normal
import numpy as np
from torch import nn
from typing import Tuple


class RandomWalker():
    def __init__(self, sigma):

        self.step_gaussian = Normal(0.0, sigma)

    def resample(self, prev_sample) -> tc.Tensor:
        return prev_sample + self.step_gaussian.sample(prev_sample.shape)


class Uniform(nn.Module):
    def __init__(self, low=0., high=1.):
        super(Uniform, self).__init__()
        self.low = tc.tensor(low) if type(low) != tc.Tensor else low
        self.high = tc.tensor(high) if type(high) != tc.Tensor else high

    def forward(self, batch_size: int = 1):
        return self.low + tc.rand(batch_size, device=self.low.device) * (self.high - self.low)

    def sample(self, batch_size: int = 1):
        return self(batch_size)


class ToProb(nn.Module):
    def forward(self, amps: tc.Tensor) -> tc.Tensor:
        return tc.exp(amps) ** 2


def initialize_samples(device, ne_atoms, atom_positions, n_samples):
    ups = []
    downs = []
    for ne_atom, atom_position in zip(ne_atoms, atom_positions):
        for e in range(ne_atom):
            if e % 2 == 0:  # fill up the orbitals alternating up down
                curr_sample_up = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                ups.append(curr_sample_up)
            else:
                curr_sample_down = np.random.normal(loc=atom_position, scale=1., size=(n_samples, 1, 3))
                downs.append(curr_sample_down)
    ups = np.concatenate(ups, axis=1)
    downs = np.concatenate(downs, axis=1)
    curr_sample = np.concatenate([ups, downs], axis=1)  # stack the ups first to be consistent with model
    curr_sample = tc.from_numpy(curr_sample.astype(np.float32)).to(device)
    return curr_sample


class MetropolisHasting(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 sampling_steps: float = 0.02,
                 correlation_length: int = 10,
                 target_acceptance = 0.5):
        super(MetropolisHasting, self).__init__()

        self.toprob = ToProb()
        self.sigma = sampling_steps
        self.model = model
        self.correlation_length = correlation_length
        self.alpha_distr = Uniform()
        self.acceptance = 0.0
        self.target_acceptance = target_acceptance
        self.adaptive = True
        print('initialized sampler')

    def forward(self, curr_samples: tc.Tensor) -> Tuple[tc.Tensor, float]:
        device = curr_samples.device
        shape = curr_samples.shape
        n_samples = shape[0]
        curr_amps = self.model(curr_samples)
        curr_probs = self.toprob(curr_amps)

        acceptance_total = 0.
        for step in range(self.correlation_length):
            # next sample
            new_samples = curr_samples + tc.normal(0.0, self.sigma, size=shape, device=device)
            new_amps = self.model(new_samples)
            new_probs = self.toprob(new_amps)

            # update sample
            alpha = new_probs / curr_probs
            mask_probs = alpha > tc.rand((n_samples,), device=device)
            mask_samples = mask_probs.unsqueeze(-1).unsqueeze(-1)

            # print('mask', mask.shape)
            curr_samples = tc.where(mask_samples, new_samples, curr_samples)
            curr_probs = tc.where(mask_probs, new_probs, curr_probs)

            acceptance_total += mask_samples.detach().sum()

        self.acceptance = float(acceptance_total / (self.correlation_length * n_samples))
        self.update_step_size()
        return curr_samples, self.acceptance

    def update_step_size(self):
        if self.acceptance < self.target_acceptance:
            self.sigma -= 0.001
        else:
            self.sigma += 0.001



if __name__ == '__main__':
    from sampling.distributions import MVGaussian
    from utils.wavefunctions import hydrogen_psi
    distr = MVGaussian(torch.zeros(3).cuda(), torch.eye(3).cuda()*2.5**2)  # square to get variance
    sampler = MetropolisHasting(hydrogen_psi, distr)
    import time
    s = time.time()
    samples, amps, acceptance = sampler.sample(1000, 100, 10)
    print(time.time() - s)
    print(samples.shape, amps.shape, acceptance)