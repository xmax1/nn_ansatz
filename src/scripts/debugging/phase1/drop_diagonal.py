


# drop diag of m x n x n x f
import jax.numpy as jnp
import numpy as np
def drop_diagonal(square):
    n = square.shape[1]
    split1 = jnp.split(square, n, axis=1)
    upper = [jnp.split(split1[i], [j], axis=2)[1] for i, j in zip(range(0, n), range(1, n))]
    lower = [jnp.split(split1[i], [j], axis=2)[0] for i, j in zip(range(1, n), range(1, n))]
    arr = [ls[i] for i in range(n-1) for ls in (upper, lower)]
    result = jnp.concatenate(arr, axis=2)
    return jnp.squeeze(result)

m, n, f = 512, 5, 10
square = jnp.array(np.random.normal(0., 1., (m, n, n, f)))
r1 = drop_diagonal(square)
mask = jnp.expand_dims(~jnp.eye(n, dtype=bool), axis=(0, 3))
mask = jnp.repeat(jnp.repeat(mask, m, axis=0), f, axis=-1)
r2 = square[mask]
r2 = r2.reshape((m, n ** 2 - n, f))
print(r1.shape, r2.shape)
print(jnp.sum(jnp.abs(r1 - r2)))
# for i, j in zip(r1[0], r2):
#     print('\n')
#     print(i, j)
