
import jax.numpy as jnp


def create_atom_batch(r_atoms, n_samples):
    return jnp.repeat(jnp.expand_dims(r_atoms, axis=0), n_samples, axis=0)