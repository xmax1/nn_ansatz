
import os
import pickle as pk


def update_dict(dictionary, name, value):
    if dictionary.get(name) is None: dictionary[name] = [value]
    else: dictionary[name].append(value)


def save_pk(x, path):
    with open(path, 'wb') as f:
        pk.dump(x, f)


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