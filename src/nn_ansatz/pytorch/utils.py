import numpy as np
import jax.numpy as jnp
import torch as tc


def update_state_dict(model_tc, params, printer=False):
    tmp = []
    for k, value in params.items():
        if k == 'intermediate':
            for intermediate in zip(*params[k]):
                for ps in intermediate:
                    tmp.append(ps)

        elif k == 'envelopes':
            order = ('linear', 'sigma', 'pi')
            for spin in (0, 1):
                for layer in order:
                    ps = params[k][layer][spin]
                    tmp.append(ps)

        else:
            tmp.append(value)

    sd = model_tc.state_dict()
    for (k, val), p in zip(sd.items(), tmp):
        print(k, val.shape, p.shape)
        assert val.shape == p.shape
        sd[k] = from_np(p)

    model_tc.load_state_dict(sd, strict=True)

    if printer:
        for k, v in model_tc.state_dict().items():
            print(k, '\n')
            print(v)

    return model_tc


def from_np(arr):
    if isinstance(arr, (np.ndarray, jnp.ndarray)):
        return tc.from_numpy(np.array(arr))
    return arr