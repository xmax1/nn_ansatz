
# import jax.numpy as jnp
import numpy as jnp


def input_activation_test(nonlinearity: str = 'sin', n_el=7, nf=3):
    split = nonlinearity.split('+')
    if 'bowl' in nonlinearity:
        bowl_features = [jnp.ones((n_el, nf))]
    else:
        bowl_features = []
    if 'sin' in nonlinearity:
        sin_desc = [x for x in split if 'sin' in x][0]
        nsin = int(sin_desc[:-3]) if len(sin_desc) > 3 else 1
        sin_features = [jnp.sin(2.*i*jnp.pi*jnp.ones((n_el, nf))) for i in range(1, nsin+1)]
    else:
        sin_features = []
    if 'cos' in nonlinearity:
        cos_desc = [x for x in split if 'cos' in x][0]
        ncos = int(cos_desc[:-3]) if len(cos_desc) > 3 else 1
        cos_features = [jnp.cos(2.*i*jnp.pi*jnp.ones((n_el, nf))) for i in range(1, ncos+1)]
    else:
        cos_features = []
    return jnp.concatenate([*sin_features, *cos_features, *bowl_features], axis=-1)


tests = ['bowl', 'sin', 'cos', 'bowl+2sin', 'bowl+2cos', '2cos', '3sin+2cos']

for test in tests:
    x = input_activation_test(nonlinearity=test)
    print(x.shape)