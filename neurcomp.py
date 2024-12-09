import subprocess
from time import perf_counter

from utils import chgcar, io2

"""
sys.argv[1] = chgcar_folder
sys.argv[2] = compress/decompress/remake
sys.argv[3] = compression_ratio
sys.argv[4] = # of layers
TODO: Add other params here
TODO: May have to fork and make basic changes to neurcomp (add option to not send in volume for net_decompress.py)
"""

NEURCOMP_DIR = "./lib/neurcomp/"

def train_func(fn: str):
    network_fn = fn.split(".")[0] + ".pth"
    config_fn = fn.split(".")[0] + ".json"
    cmd = ["python", NEURCOMP_DIR + "train.py",
            "--volumne", fn,
            "--network", network_fn,
            "--config", config_fn,
            "--d_out", "3",
            "--compression_ratio", str(sys.argv[3]),
            "--n_layers", str(sys.argv[4])]

    time_start = perf_counter()
    subprocess.run(cmd)
    time_end = perf_counter()

    return network_fn, config_fn, time_end - time_start


def compress_func(network_fn: str, config_fn: str):
    compresed_fn = network_fn.split(".")[0] + "_compressed"

    cmd = ["python", NEURCOMP_DIR + "compress.py",
        "--net", network_fn,
        "--config", config_fn,
        "--compressed", compresed_fn]

    time_start = perf_counter()
    subprocess.run(cmd)
    time_end = perf_counter()

    return compresed_fn, time_end - time_start


def decompress_func(compresed_fn: str):





