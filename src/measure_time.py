from nn_ansatz import setup, measure_kfac_and_energy_time
import pandas as pd

print('pre loop')
n_walkers = 2048
atol = 1e-5
n_els = range(2, 40, 2)

times = []
for n_el in n_els:
    print('trying n_el %i' % n_el)
    n_up = n_el // 2
    # n_up = n_el

    cfg = setup(n_it=1000, atol=atol, n_el=n_el, n_up=n_up, n_walkers=n_walkers, density_parameter=1., name='timing', pretrain=False)
    cfg = measure_kfac_and_energy_time(cfg)
    t = [n_el, n_up, cfg['sampling_time'], cfg['kfac_time']]
    times.append(t)
    print(t)

    times_save = pd.DataFrame(times, columns=['n_el', 'n_up', 'sampling_time', 'kfac_time'])
    times_save.to_csv('/home/energy/amawi/projects/nn_ansatz/src/exp/times/2201/8gpu_para_kfac.csv', index=False)