
# curr_walkers = rnd.uniform(key, (n_walkers_max, n_el, 3), minval=0., maxval=max_distance*2)
# curr_prob = jnp.exp(vwf(params, curr_walkers))**2


# def hack_sample(
#     curr_walkers, 
#     curr_prob,
#     key, max_distance, params, vwf, basis, inv_basis
#     ):

#     key, *subkeys = rnd.split(key, 5)
#     shape = curr_walkers.shape
#     # uni_walkers = rnd.uniform(subkey, shape, minval=0., maxval=max_distance*2)
#     # print(uni_walkers.shape)
#     # prob_uni = jnp.exp(vwf(params, uni_walkers))**2

#     curr_walkers, acc_met, step_size = sampler(params, curr_walkers, subkeys[0], 0.01)
#     curr_prob = jnp.exp(vwf(params, curr_walkers))**2

#     mean = rnd.uniform(subkeys[0], minval=0., maxval=max_distance*2)
#     std = rnd.uniform(subkeys[1], minval=0.01, maxval=4)
#     new_walkers = (rnd.normal(subkeys[3], shape) + mean) * std
#     new_walkers = keep_in_boundary(new_walkers, basis, inv_basis)
#     new_prob = jnp.exp(vwf(params, new_walkers))**2

#     alpha = new_prob / curr_prob
#     new_mask = alpha > rnd.uniform(subkeys[2], (shape[0],))
#     curr_prob = jnp.where(new_mask, new_prob, curr_prob)
#     curr_walkers = jnp.where(new_mask[:, None, None], new_walkers, curr_walkers)

#     acc = jnp.mean(new_mask)
#     return curr_prob, curr_walkers, new_mask, acc, acc_met
    
# _hack_sample = jit(partial(hack_sample, max_distance=max_distance, params=params, vwf=vwf, basis=mol.basis, inv_basis=mol.inv_basis))

# n_steps = 100000
# t0 = time()
# e_every = 100
# for step in range(n_steps):
#     key, subkey = rnd.split(key)
#     curr_prob, curr_walkers, new_mask, acc, acc_met = _hack_sample(curr_walkers, curr_prob, subkey)
#     # tag = f'% norm > uni: {jnp.mean(mask)}' 
#     tag = f' % new: {jnp.mean(new_mask):.3f}'
#     tag += f' mean prob: {jnp.mean(curr_prob):.3E}'
#     tag += f' acc: {jnp.mean(acc):.3f}'
#     tag += f' acc met: {jnp.mean(acc_met):.3f}'
#     t0 = track_time(step, n_steps, t0, tag=tag, every_n=e_every)
#     if step % e_every == 0:
#         pe, ke = compute_energy(params, curr_walkers)
#         print(f'Step: {step} | E: {jnp.mean(ke + pe):.6f}')



if input_bool(args['train']):

    from optax import adam, apply_updates
    from nn_ansatz.vmc import  create_grad_function
    from nn_ansatz.utils import load_pk, key_gen, split_variables_for_pmap
    from nn_ansatz.parameters import initialise_params

    w = rnd.uniform(key, (n_walkers_max, n_el, 3), minval=0., maxval=max_distance*2)
    grad_fn = create_grad_function(mol, vwf)
    p = initialise_params(mol, key)
    optimizer = adam(cfg['lr'])
    state = optimizer.init(p)
    energy_function = create_energy_fn(mol, vwf, separate=True)
    step_size = 0.02
    _apply_updates = jit(apply_updates)

    print('entering train loop')
    n_steps = 1000
    t0 = time()
    for step in range(n_steps):
        key, subkey = rnd.split(key)
        w, acceptance, step_size = sampler(p, w, subkey, step_size)
        grads, e_locs = grad_fn(p, w, jnp.array([step]))
        grads, state = optimizer.update(grads, state)
        p = _apply_updates(p, grads)
        
        tag = f' E: {jnp.mean(e_locs):.3f} +- {jnp.std(e_locs):.3f}'
        tag += f' acc: {jnp.mean(acceptance):.3f}'
        tag += f' step_size: {jnp.mean(step_size):.3f}'
        t0 = track_time(step, n_steps, t0, tag=tag, every_n=1)

        if step == 100:
            save_pk(np.array(w), run_dir / 'tr_walker_i100.pk')

    walkers = w
    save_pk(np.array(w), run_dir / 'tr_walker_i1000.pk')
    

if equilibrate:
    print('EQUILIBRATING WALKERS')
    walkers = rnd.uniform(key, (n_walkers_max, n_el, 3), minval=0., maxval=max_distance*2)
    # walkers = jnp.array(load_pk('/home/energy/amawi/projects/nn_ansatz/src/experiments/PRX_Responses/runs/run41035/eq_walkers_i100000.pk'))
    _ = sampler(params, walkers, key, 0.02)
    walkers = equilibrate_1M_walkers(key, n_walkers_max, walkers[:n_walkers_max])
    save_pk(np.array(walkers), equilibrated_walkers_path)
else:
    print('LOADING WALKERS') 
    walkers = jnp.array(load_pk(equilibrated_walkers_path))
