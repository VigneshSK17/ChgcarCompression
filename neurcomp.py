import subprocess
import sys
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
    compressed_fn = network_fn.split(".")[0] + "_neurcomp_compressed"

    cmd = ["python", NEURCOMP_DIR + "net_compress.py",
        "--net", network_fn,
        "--config", config_fn,
        "--compressed", compressed_fn]

    time_start = perf_counter()
    subprocess.run(cmd)
    time_end = perf_counter()

    compressed_fs = io2.get_file_size_mb(compressed_fn)

    return compressed_fn, compressed_fs, time_end - time_start


def decompress_func(compressed_fn: str, volume_fn: str = None):
    decompressed_fn = compressed_fn.split("_compressed")[0] + "_decompressed"

    cmd = ["python", NEURCOMP_DIR + "net_decompress.py",
        "--compressed", compressed_fn,
        "--recon", decompressed_fn]

    time_start = perf_counter()
    subprocess.run(cmd)
    time_end = perf_counter()

    return decompressed_fn + ".vti", time_end - time_start


def retrieve_compressed(file: str):
    chgcar_fn = file.split("_neurcomp_compressed")[0]
    files_required = [f"{chgcar_fn}_neurcomp_compressed", f"{chgcar_fn}_dims.txt", f"{chgcar_fn}_structure.cif", f"{chgcar_fn}_data_aug.txt"]
    if not io2.check_files(files_required):
        print("f{file}: Missing files for decompression")
        return None

    structure, lattice, data_aug, dims = chgcar.retrieve_structure_aug_dims_pymatgen(chgcar_fn)

    return file, dims, structure, lattice, data_aug


def compress_file_helper(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)

    charge = charge_pgrid.grid_data
    mag = mag_pgrid.grid_data

    charge_network_fn, charge_config_fn, charge_train_duration = train_func(charge)
    mag_network_fn, mag_config_fn, mag_train_duration = train_func(mag)

    charge_compressed_fn, charge_fs, charge_compress_duration = compress_func(charge_network_fn, charge_config_fn)
    mag_compressed_fn, mag_fs, mag_compress_duration = compress_func(mag_network_fn, mag_config_fn)

    chgcar.store_structure_aug_dims_pymatgen(file_no_ext, structure, data_aug, dims)

    total_train_duration = charge_train_duration + mag_train_duration
    total_compress_duration = charge_compress_duration + mag_compress_duration

    return file_no_ext, charge_pgrid, mag_pgrid, charge_train_duration, f"{total_train_duration},{total_compress_duration}", fs, charge_fs, mag_fs


def decompress_file_helper(file: str):
    # TODO: Make sure to get right _compressed file
