
import subprocess
import re
import sys
sys.path.append('..')
from nn_ansatz.python_helpers import submit_job_to_any_gpu

script_path = '/home/amawi/projects/nn_ansatz/src/run_with_args.py'

# cmd = 'python ' + script_path + ' ' + exp
# exp = '-s {} -n_sh {} -n_ph {} -orb {} -n_el {} -inact {} -dp {} -n_det {} -name {} -n_it'\
#                        .format(system, n_sh, n_ph, orbitals, n_el, inact, dp, n_det, name, n_it)
# exp_cmd = '-s {} -n_sh {} -n_ph {} -orb {} -n_el {} -inact {} -dp {} -n_det {} -name {} -n_it'\
#                        .format(**exp)

hosts = ['titan02']
inacts = ['bowl', 'sin', 'sin+cos', 'sin+cos+bowl', '2sin+2cos', '3sin+3cos']
inacts = ['']
processes = []
for inact in inacts:
    exp = ' -s {} -n_sh {} -n_ph {} -orb {} -n_el {} -inact {} -dp {} -n_det {} -name {} -n_it {}'\
                       .format('HEG', 64, 16, 'real_plane_waves', 7, inact, 1.0, 1, '101121/inact_sweep_smoothdist', 50000)
    cmd = 'python ' + script_path + exp
    x = submit_job_to_any_gpu(cmd, hosts=hosts)
    processes.append(x)



