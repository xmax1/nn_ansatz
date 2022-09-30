import subprocess
import os

machines = range(0, 8)
factors = [(i+1)/2. for i in machines]

for i, factor in zip(machines, factors):
    os.environ['CUDA_VISIBLE_DEVICES'] = '%i' % i
    subprocess.Popen(['python',  'run_vmc.py', '%.2f' % factor])