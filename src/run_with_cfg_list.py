import argparse
from nn_ansatz import setup, run_vmc, approximate_energy
from distutils.util import strtobool
from nn_ansatz import load_pk
import tensorflow as tf
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--cfg_path", type=str, default='')
    args = parser.parse_args()
    return args

args = get_args()

tf.config.experimental.set_visible_devices([], "GPU")
os.environ['DISTRIBUTE'] = 'True'

cfg = load_pk(args.cfg_path)
# log = run_vmc(cfg)

for load_it in range(10000, cfg['n_it']+1, 10000):
    approximate_energy(cfg, load_it=load_it)
