from collections import defaultdict
import json
import resource
import sys
import subprocess
from time import perf_counter

from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

from utils import chgcar, io, io2

TTHRESH_BIN = "./lib/tthresh/build/tthresh"

def main():
    folder = sys.argv[1]
    method = sys.argv[2]
    # error_bound, error_value = sys.argv[3], sys.argv[4]

    if not io2.check_dir(folder):
        print("Invalid directory")
        sys.exit(1)

    files = io2.get_files_in_dir(folder)

    if method == "compress":
        orig_values, all_metrics = io2.compress_dir(files, compress_file_helper, "tthresh", write=True)
        for file_no_ext, file_metrics in all_metrics.items():
            print(file_no_ext, "Compression Duration: ", file_metrics["compress_duration"], "s")
            print(file_no_ext, "Charge Original File Size: ", file_metrics["orig_file_size"], "MB")
            print(file_no_ext, "Charge Compressed Data Size: ", file_metrics["compressed_data_size"], "MB")

    if method == "decompress":
        # decompressed_values = io.decompress_dir(files, decompress_func)
        # for file_name, (charge, mag) in decompressed_values.items():
        #     print(file_name, charge.shape, mag.shape)
        decompressed_values, all_metrics = io2.decompress_dir(files, decompress_file_helper, "tthresh")
        for file_no_ext, file_metrics in all_metrics.items():
            print(file_no_ext, "Decompression Duration: ", file_metrics["decompress_duration"], "s")

    if method == "remake":
        print("Starting compression...")
        orig_values, compress_metrics = io2.compress_dir(files, compress_file_helper, "pyrho")
        print("Starting decompression...")
        decompressed_values, decompress_metrics = io2.decompress_dir(files, decompress_file_helper, "pyrho")

        all_metrics = chgcar.generate_metrics(orig_values, decompressed_values, compress_metrics, decompress_metrics)
        print(json.dumps(all_metrics, sort_keys=True, indent=4))

    if method == "remake_no_file":
        print("Starting compression...")
        orig_values, _, compress_metrics = io2.compress_dir(files, compress_data, "pyrho", write=False)
        print("Starting decompression...")
        decompressed_values, decompress_metrics = io2.decompress_dir_no_file(orig_values, decompress_data, has_data=False)

        all_metrics = chgcar.generate_metrics(orig_values, decompressed_values, compress_metrics, decompress_metrics)
        print(json.dumps(all_metrics, sort_keys=True, indent=4))

def compress_data(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, _ = chgcar.parse_chgcar_pymatgen(file)

    charge = charge_pgrid.grid_data
    mag = mag_pgrid.grid_data

    chgcar.data_to_raw(charge, dims, f"{file_no_ext}_tthresh_charge.raw")
    chgcar.data_to_raw(mag, dims, f"{file_no_ext}_tthresh_mag.raw")

    charge_compress_duration = compress_func(file_no_ext, "charge", dims)
    mag_compress_duration = compress_func(file_no_ext, "mag", dims)

    return file_no_ext, structure, charge_pgrid, mag_pgrid, data_aug, dims, None, None, charge_compress_duration + mag_compress_duration

def compress_file_helper(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)

    charge = charge_pgrid.grid_data
    mag = mag_pgrid.grid_data

    chgcar.data_to_raw(charge, dims, f"{file_no_ext}_tthresh_charge.raw")
    chgcar.data_to_raw(mag, dims, f"{file_no_ext}_tthresh_mag.raw")

    charge_compress_duration = compress_func(file_no_ext, "charge", dims)
    mag_compress_duration = compress_func(file_no_ext, "mag", dims)

    io2.delete_files([f"{file_no_ext}_tthresh_charge.raw", f"{file_no_ext}_tthresh_mag.raw"])

    charge_fs = io2.get_file_size_mb(f"{file_no_ext}_tthresh_charge_compressed.raw")
    mag_fs = io2.get_file_size_mb(f"{file_no_ext}_tthresh_mag_compressed.raw")

    chgcar.store_structure_aug_dims_pymatgen(file_no_ext, structure, data_aug, dims)

    return file_no_ext, charge_pgrid, mag_pgrid, charge_compress_duration + mag_compress_duration, fs, charge_fs, mag_fs

def compress_func(chgcar_fn: str, section: str, dims: list[int]):
    time_start = perf_counter()
    cmd = get_tthresh_compress_cmd(chgcar_fn, section, dims)
    res = subprocess.run(cmd)
    time_end = perf_counter()

    return time_end - time_start


def decompress_func(compressed_fn: str):
    file_no_ext = compressed_fn.split(".")[0]

    time_start = perf_counter()
    cmd = get_tthresh_decompress_cmd(file_no_ext)
    subprocess.run(cmd)
    time_end = perf_counter()

    return time_end - time_start

def decompress_data(file_no_ext, charge, mag, lattice, dims):
    decompress_charge_duration = decompress_func(f"{file_no_ext}_tthresh_charge_compressed.raw")
    decompress_mag_duration = decompress_func(f"{file_no_ext}_tthresh_mag_compressed.raw")

    decompress_charge = chgcar.raw_to_data(f"{file_no_ext}_tthresh_charge_compressed_decompressed.raw")
    decompress_mag = chgcar.raw_to_data(f"{file_no_ext}_tthresh_mag_compressed_decompressed.raw")

    decompress_charge = decompress_charge.reshape(dims)
    decompress_mag = decompress_mag.reshape(dims)

    decompressed_charge_pgrid, decompressed_mag_pgrid = PGrid(decompress_charge, lattice), PGrid(decompress_mag, lattice)

    return file_no_ext, decompressed_charge_pgrid, decompressed_mag_pgrid, decompress_charge_duration + decompress_mag_duration

def decompress_file_helper(file: str):
    chgcar_fn = io2.get_only_file_name(file)
    files_required = [f"{chgcar_fn}_tthresh_charge_compressed.raw", f"{chgcar_fn}_tthresh_mag_compressed.raw", f"{chgcar_fn}_structure.cif", f"{chgcar_fn}_data_aug.txt", f"{chgcar_fn}_dims.txt"]
    if not io2.check_files(files_required):
        print(f"{file}: Missing files for decompression")
        return None

    decompress_charge_duration = decompress_func(f"{chgcar_fn}_tthresh_charge_compressed.raw")
    decompress_mag_duration = decompress_func(f"{chgcar_fn}_tthresh_mag_compressed.raw")

    decompress_charge = chgcar.raw_to_data(f"{chgcar_fn}_tthresh_charge_compressed_decompressed.raw")
    decompress_mag = chgcar.raw_to_data(f"{chgcar_fn}_tthresh_mag_compressed_decompressed.raw")
    structure, lattice, data_aug, dims = chgcar.retrieve_structure_aug_dims_pymatgen(chgcar_fn)

    decompress_charge = decompress_charge.reshape(dims)
    decompress_mag = decompress_mag.reshape(dims)

    charge_pgrid, mag_pgrid = PGrid(decompress_charge, lattice), PGrid(decompress_mag, lattice)

    return chgcar_fn, structure, data_aug, charge_pgrid, mag_pgrid, decompress_charge_duration + decompress_mag_duration


def get_tthresh_compress_cmd(chgcar_fn: str, section: str, dims: list[int]):
    cmd =  [TTHRESH_BIN,
                    "-i", f"{chgcar_fn}_tthresh_{section}.raw",
                    "-t", "double",
                    "-s", str(dims[0]), str(dims[1]), str(dims[2]),
                    sys.argv[3], sys.argv[4],
                    "-c", f"{chgcar_fn}_tthresh_{section}_compressed.raw"]
    return cmd

def get_tthresh_decompress_cmd(compressed_fn: str):
    cmd =  [TTHRESH_BIN,
                    "-c", f"{compressed_fn}.raw",
                    "-o", f"{compressed_fn}_decompressed.raw"]
    return cmd

if __name__ == "__main__":
    main()