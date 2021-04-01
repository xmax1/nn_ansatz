

import torch as tc
import jax.numpy as jnp


def compare(tc_arr, jnp_arr):
    arr = tc_arr.detach().cpu().numpy()
    diff = arr - jnp_arr
    print('l1-norm: ', jnp.mean(jnp.abs(diff)))