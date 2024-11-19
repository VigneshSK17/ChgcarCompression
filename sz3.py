import gzip
import sys
from time import perf_counter
import numpy as np
from pyrho.charge_density import ChargeDensity, PGrid

from lib.sz3.tools.pysz import pysz
from utils import chgcar, io2

"""
sys.argv[1] = relative target accuracy
"""

SZPATH = "lib/sz3/build/tools/sz3c/libSZ3c.dylib"
SZ3 = pysz.SZ(SZPATH)

# sz = pysz.SZ(SZPATH)

def compress_func(charge: np.ndarray, mag: np.ndarray):
    time_start = perf_counter()

    charge_compressed_data, _ = SZ3.compress(charge, 1, None, sys.argv[1], None)
    mag_compressed_data, _ = SZ3.compress(mag, 1, None, sys.argv[1], None)

    time_end = perf_counter()

    return charge_compressed_data, mag_compressed_data, time_end - time_start


def store_compressed(chgcar_fn: str, charge, mag, structure, data_aug, dims):
    with gzip.GzipFile(f"{chgcar_fn}_sz3_compressed_charge.npy.gz", "w") as fc:
        np.save(fc, charge)
    with gzip.GzipFile(f"{chgcar_fn}_sz3_compressed_mag.npy.gz", "w") as fm:
        np.save(fm, mag)

    charge_fs = io2.get_file_size_mb(f"{chgcar_fn}_sz3_compressed_charge.npy.gz")
    mag_fs = io2.get_file_size_mb(f"{chgcar_fn}_sz3_compressed_mag.npy.gz")

    chgcar.store_structure_aug_dims_pymatgen(chgcar_fn, structure, data_aug, dims)

    return charge_fs, mag_fs


def retrieve_compressed(file: str):
    chgcar_fn = io2.get_only_file_name(file)
    files_required = [f"{chgcar_fn}_sz3_compressed_charge.npy.gz", f"{chgcar_fn}_sz3_compressed_mag.npy.gz", f"{chgcar_fn}_dims.txt", f"{chgcar_fn}_structure.cif", f"{chgcar_fn}_data_aug.txt"]
    if not io2.check_files(files_required):
        print(f"{file}: Missing files for decompression")
        return None

    with gzip.GzipFile(f"{chgcar_fn}_sz3_compressed_charge.npy.gz", "r") as fc:
        charge_compressed = np.load(fc)
    with gzip.GzipFile(f"{chgcar_fn}_sz3_compressed_mag.npy.gz", "r") as fm:
        mag_compressed = np.load(fm)

    structure, lattice, data_aug, dims = chgcar.retrieve_structure_aug_dims_pymatgen(chgcar_fn)

    return chgcar_fn, charge_compressed, mag_compressed, dims, structure, lattice, data_aug


def decompress_func(charge: np.ndarray, mag: np.ndarray, dims: list[int]):
    time_start = perf_counter()

    charge_decompressed_data = SZ3.decompress(charge, dims, np.float32)
    mag_decompressed_data = SZ3.decompress(mag, dims, np.float32)

    time_end = perf_counter()

    return charge_decompressed_data, mag_decompressed_data, time_end - time_start


def compress_data(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)

    charge_compressed, mag_compressed, compress_duration = compress_func(charge_pgrid, mag_pgrid)

    return file_no_ext, charge_pgrid, mag_pgrid, data_aug, dims, charge_compressed, mag_compressed, compress_duration


def compress_file_helper(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)

    charge = charge_pgrid.grid_data
    mag = mag_pgrid.grid_data

    charge_compressed, mag_compressed, compress_duration = compress_func(charge, mag)
    charge_fs, mag_fs = store_compressed(file_no_ext, charge_compressed, mag_compressed, structure, data_aug, dims)

    return file_no_ext, charge_pgrid, mag_pgrid, compress_duration, fs, charge_fs, mag_fs
