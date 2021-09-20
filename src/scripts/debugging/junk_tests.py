
from jax import grad
import jax.numpy as jnp

norm_grad = grad(jnp.linalg.norm)
x = norm_grad(999999999999999.)
print(x)

def norm1(x):
    y = x**2
    return jnp.sqrt(y)

norm_grad = grad(norm1)
x = norm_grad(jnp.inf)
print(x)

def norm2(x):
    return jnp.sqrt(x)

print(grad(norm2)(jnp.inf**2))