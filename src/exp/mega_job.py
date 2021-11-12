
import subprocess
import re
import sys
sys.path.append('/home/amawi/projects/nn_ansatz/src')
from nn_ansatz.python_helpers import submit_job_to_any_gpu

script_path = '/home/amawi/projects/nn_ansatz/src/run_with_args.py'

# BASE STATS
system = 'LiSolidBCC'
n_sh = 64
n_ph = 16
orbitals = 'real_plane_waves'
n_el = 7
inact = 'sin+cos+bowl'
density_parameter = 1.
n_det = 1
name = '1211/isotropic_comparison'
n_it = 50000

hosts = ['titan02']
hypams = ['bowl', 'sin', 'sin+cos', 'sin+cos+bowl', '2sin+2cos', '3sin+3cos']
hypams = ['isotropic_spline', 'isotropic_sphere']
processes = []
for orbitals in hypams:
    exp = ' -s {system} -n_sh {n_sh} -n_ph {n_ph} -orb {orbitals} -n_el {n_el} -inact {inact} -dp {density_parameter} -n_det {n_det} -name {name} -n_it {n_it}'\
                       .format(system=system, 
                               n_sh=n_sh, 
                               n_ph=n_ph,
                               orbitals=orbitals, 
                               n_el=n_el, 
                               inact=inact, 
                               density_parameter=density_parameter, 
                               n_det=n_det, 
                               name=name, 
                               n_it=n_it)

    cmd = 'python ' + script_path + exp
    x = submit_job_to_any_gpu(cmd, hosts=hosts)
    processes.append(x)



