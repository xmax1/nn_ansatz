
import os
import pickle as pk
import subprocess
import re

import xml.etree.ElementTree as ET
from time import sleep

# cmd = 'nvidia-smi --query-gpu=memory.free --format=csv'
# subprocess.Popen("ssh {user}@{host}".format(user=user, host=host, cmd=cmd), \
#     shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()

HOSTS = ['titan01', 'titan02', 'titan04', 'titan05', 'titan07', 'titan10', 'titan11', 'titan12']
user = 'amawi'
nvidia_cmd = 'nvidia-smi --query-gpu=memory.free --format=csv'

def submit_job_to_any_gpu(cmd, hosts=None):
    if hosts is None: hosts = HOSTS
    executed=False
    while not executed:
        for host in hosts:
            mems = str(subprocess.Popen("ssh {user}@{host} {cmd}".format(user=user, host=host, cmd=nvidia_cmd), \
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0])
            mems = re.findall("\d+", mems)
            free_mems = [m for m in mems if int(m) > 9000]
            if len(free_mems) > 0:
                gpu = mems.index(free_mems[0])
                print('host %s node spaces %s chosen gpu %i' % (host, mems, gpu))   
                cmd = ('CUDA_VISIBLE_DEVICES=%i '+ cmd) % gpu  # ' >null 2>&1' pipes the outputs to other places so the ssh can close
                print(cmd)
                # out = str(subprocess.Popen("ssh {user}@{host} {cmd}".format(user=user, host=host, cmd=cmd)), shell=True)
                # 'screen -dmS exp bash -c "ssh titan08; sleep 10"'
                # out = subprocess.run('screen -dmS exp bash -c "ssh {user}@{host}; {cmd}"'.format(user=user, host=host, cmd=cmd), shell=True)
                # out = subprocess.run('screen -dmS exp bash -c "{cmd}"'.format(cmd=cmd), shell=True)
                x = subprocess.run('screen -dmS exp%i bash -c "%s"' % (gpu, cmd), shell=True)
                sleep(30)
                return x
        sleep(60)
        print('waiting for a gpu')
    

def update_dict(dictionary, name, value):
    if dictionary.get(name) is None: dictionary[name] = [value]
    else: dictionary[name].append(value)


def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)


def find_all_files_in_dir(root, name):
    import os
    all_files = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file == name:
                all_files.append(os.path.join(root, file))
    return all_files


def load_pk(path):
    with open(path, 'rb') as f:
        x = pk.load(f)
    return x


def make_dir(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def join_and_create(*args):
    path = os.path.join(*args)
    if not os.path.exists(path): os.makedirs(path)
    return path