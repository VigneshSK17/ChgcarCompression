from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

from utils import chgcar, io, io2

"""
sys.argv[1] = chgcar_folder
sys.argv[2] = compress/decompress
sys.argv[3] = dims_divisor
sys.argv[4] = smear_std
"""

# TODO: Time using timeit.timeit not resource.getrusage
def compress_func(charge: PGrid, mag: PGrid, dims: list[int]):

    time_start = perf_counter()

    dims = charge.grid_shape
    compressed_dims = [dim // int(sys.argv[3]) for dim in dims]

    charge_compressed = charge.lossy_smooth_compression(compressed_dims, float(sys.argv[4]))
    mag_compressed = mag.lossy_smooth_compression(compressed_dims, float(sys.argv[4]))

    time_end = perf_counter()

    return charge_compressed, mag_compressed, time_end - time_start


def store_compressed(chgcar_fn: str, charge, mag, structure, data_aug, dims):
    # np.savez_compressed(f"{chgcar_fn}_compressed.npz", charge=charge, mag=mag, dims=dims)
    with gzip.GzipFile(f"{chgcar_fn}_pyrho_compressed_charge.npy.gz", "w") as fc:
        np.save(fc, charge)
    with gzip.GzipFile(f"{chgcar_fn}_pyrho_compressed_mag.npy.gz", "w") as fm:
        np.save(fm, mag)

    charge_fs = io2.get_file_size_mb(f"{chgcar_fn}_pyrho_compressed_charge.npy.gz")
    mag_fs = io2.get_file_size_mb(f"{chgcar_fn}_pyrho_compressed_mag.npy.gz")

    chgcar.store_structure_aug_dims_pymatgen(chgcar_fn, structure, data_aug, dims)

    return charge_fs, mag_fs

def retrieve_compressed(file: str):
    chgcar_fn = io2.get_only_file_name(file)
    files_required = [f"{chgcar_fn}_pyrho_compressed_charge.npy.gz", f"{chgcar_fn}_pyrho_compressed_mag.npy.gz", f"{chgcar_fn}_dims.txt", f"{chgcar_fn}_structure.cif", f"{chgcar_fn}_data_aug.txt"]
    if not io2.check_files(files_required):
        print(f"{file}: Missing files for decompression")
        return None

    with gzip.GzipFile(f"{chgcar_fn}_pyrho_compressed_charge.npy.gz", "r") as fc:
        charge_compressed = np.load(fc)
    with gzip.GzipFile(f"{chgcar_fn}_pyrho_compressed_mag.npy.gz", "r") as fm:
        mag_compressed = np.load(fm)

    structure, lattice, data_aug, dims = chgcar.retrieve_structure_aug_dims_pymatgen(chgcar_fn)

    return chgcar_fn, charge_compressed, mag_compressed, dims, structure, lattice, data_aug


def decompress_func(data: np.ndarray, lattice: np.ndarray, dims: list[int]):
    time_start = perf_counter()

    pgrid = PGrid(data, lattice)
    up_sample_ratio = dims[0] // data.shape[0]
    pgrid_upscale = pgrid.get_transformed(
        sc_mat=np.eye(3),
        grid_out=dims,
        up_sample=up_sample_ratio
        # up_sample=1
    )

    time_end = perf_counter()

    return pgrid_upscale, time_end - time_start

def remake_chgcar(chgcar_fn: str, charge_pgrid: PGrid, mag_pgrid: PGrid, structure: Structure, data_aug):

    cgden = ChargeDensity(pgrids={"total": charge_pgrid, "diff": mag_pgrid}, structure=structure)

    chgcar = cgden.to_Chgcar()
    chgcar.data_aug = data_aug
    chgcar.write_file(f"{chgcar_fn}_pyrho.vasp")

def compress_data(file: str, file_no_ext: str):
    structure, charge, mag, data_aug, dims, _ = chgcar.parse_chgcar_pymatgen(file)
    charge_compressed, mag_compressed, compress_duration = compress_func(charge, mag, dims)

    return file_no_ext, structure, charge, mag, data_aug, dims, charge_compressed, mag_compressed, compress_duration

def compress_file_helper(file: str, file_no_ext: str):
    structure, charge, mag, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)

    charge_compressed, mag_compressed, compress_duration = compress_func(charge, mag, dims)

    compressed_charge_fs, compressed_mag_fs = store_compressed(file_no_ext, charge_compressed, mag_compressed, structure, data_aug, dims)

    return file_no_ext, charge, mag, compress_duration, fs, compressed_charge_fs, compressed_mag_fs

def decompress_data(file_no_ext, charge, mag, lattice, dims):
    decompress_charge, decompress_charge_duration = decompress_func(charge, lattice, dims)
    decompress_mag, decompress_mag_duration = decompress_func(mag, lattice, dims)

    return file_no_ext, decompress_charge, decompress_mag, decompress_charge_duration + decompress_mag_duration

def decompress_file_helper(file: str):
    if retrieve_compressed(file) is None:
        return None
    chgcar_fn, charge_compressed, mag_compressed, dims, structure, lattice, data_aug = retrieve_compressed(file)

    _, decompress_charge, decompress_mag, decompress_duration = decompress_data(chgcar_fn, charge_compressed, mag_compressed, lattice, dims)

    return chgcar_fn, structure, data_aug, decompress_charge, decompress_mag, decompress_duration


def main():
    folder = sys.argv[1]
    method = sys.argv[2]

    if not io.check_dir(folder):
        print("Invalid directory")
        sys.exit(1)
    files = io.get_files_in_dir(folder)

    if method == "compress":
        orig_values, all_metrics = io2.compress_dir(files, compress_file_helper, "pyrho")
        for file_no_ext, file_metrics in all_metrics.items():
            print(file_no_ext, "Compression Duration: ", file_metrics["compress_duration"], "s")
            print(file_no_ext, "Charge Original File Size: ", file_metrics["orig_file_size"], "MB")
            print(file_no_ext, "Charge Compressed Data Size: ", file_metrics["compressed_data_size"], "MB")

    if method == "decompress":
        decompressed_values, all_metrics = io2.decompress_dir(files, decompress_file_helper, "pyrho")
        for file_no_ext, file_metrics in all_metrics.items():
            print(file_no_ext, "Decompression Duration: ", file_metrics["decompress_duration"], "s")

    # Compresses and then decompresses CHGCAR's, provides MAE
    if method == "remake":
        print("Starting compression...")
        orig_values, compress_metrics = io2.compress_dir(files, compress_file_helper, "pyrho")
        print("Starting decompression...")
        decompressed_values, decompress_metrics = io2.decompress_dir(files, decompress_file_helper, "pyrho")

        all_metrics = chgcar.generate_metrics(orig_values, decompressed_values, compress_metrics, decompress_metrics)
        chgcar.write_metrics_to_file("metrics.json", all_metrics, f"pyrho_{sys.argv[3]}_{sys.argv[4]}")

    if method == "remake_no_file":
        print("Starting compression...")
        orig_values, compressed_values, compress_metrics = io2.compress_dir(files, compress_data, "pyrho", write=False)
        print("Starting decompression...")
        decompressed_values, decompress_metrics = io2.decompress_dir_no_file(compressed_values, decompress_data)

        all_metrics = chgcar.generate_metrics(orig_values, decompressed_values, compress_metrics, decompress_metrics)
        print(json.dumps(all_metrics, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()