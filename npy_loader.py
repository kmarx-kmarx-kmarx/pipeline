import numpy as np
import gcsfs
import os


def load_npy(filepath, id, fs=None, root_remote=False, save_local=False, dest_dir=None):
    if root_remote:
        if save_local:
            if not os.path.exists(os.path.join(dest_dir, f"{id}_spots.npy")):
                fs.get(filepath, os.path.join(dest_dir, f"{id}_spots.npy"))
            arr = np.load(os.path.join(dest_dir, f"{id}_spots.npy"))
        else:
            with fs.open(filepath, 'rb') as f:
                arr = np.load(f)
    else:
        arr = np.load(filepath)

    return arr
