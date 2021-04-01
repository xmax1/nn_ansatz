
import jax
import jax.numpy as jnp
from jax import lax

def local_kinetic_energy(f):

  def _lapl_over_f(params, x):  # this is recalled everytime
    n = x.shape[0]
    eye = jnp.eye(n)
    grad_f = jax.grad(f, argnums=1)
    grad_f_closure = lambda y: grad_f(params, y)  # ensuring the input can be just x

    def _body_fun(i, val):
      # primal is the first order evaluation
      # tangent is the second order
      primal, tangent = jax.jvp(grad_f_closure, (x,), (eye[i],))
      return val + primal[i]**2 + tangent[i]

    # from lower to upper
    # (lower, upper, func(int, a) -> a, init_val)
    # this is like functools.reduce()
    # val is the previous  val (initialised to 0.0)
    return -0.5 * lax.fori_loop(0, n, _body_fun, 0.0)

  return _lapl_over_f


def potential_energy(r_ae, r_ee, atoms, charges):
  """Returns the potential energy for this electron configuration.
  Args:
    r_ae: Shape (nelectrons, natoms). r_ae[i, j] gives the distance between
      electron i and atom j.
    r_ee: Shape (neletrons, nelectrons, :). r_ee[i,j,0] gives the distance
      between electrons i and j. Other elements in the final axes are not
      required.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  """

  v_ee = jnp.sum(jnp.triu(1 / r_ee[..., 0], k=1))
  v_ae = -jnp.sum(charges / r_ae[..., 0])  # pylint: disable=invalid-unary-operand-type
  r_aa = jnp.linalg.norm(atoms[None, ...] - atoms[:, None], axis=-1)
  v_aa = jnp.sum(jnp.triu((charges[None, ...] * charges[..., None]) / r_aa, k=1))
  return v_ee + v_ae + v_aa


def compute_local_energy(f, atoms, charges):
  """Creates function to evaluate the local energy.
  Args:
    f: Callable with signature f(data, params) which returns the log magnitude
      of the wavefunction given parameters params and configurations data.
    atoms: Shape (natoms, ndim). Positions of the atoms.
    charges: Shape (natoms). Nuclear charges of the atoms.
  Returns:
    Callable with signature e_l(params, data) which evaluates the local energy
    of the wavefunction given the parameters params and a single MCMC
    configuration in data.
  """
  ke = local_kinetic_energy(f)

  def _e_l(params, x):
    """Returns the total energy.
    Args:
      params: network parameters.
      x: MCMC configuration.
    """
    _, _, r_ae, r_ee = networks.construct_input_features(x, atoms)
    potential = potential_energy(r_ae, r_ee, atoms, charges)
    kinetic = ke(params, x)
    return potential + kinetic

  return _e_l