from collections import defaultdict
import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import pathlib

from utils import chgcar

def compress_dir(files: list[str], compress_file_func, compressor_name: str, write = True):

    orig_values = {}
    compressed_values = {}
    metrics = defaultdict(dict)

    with ThreadPoolExecutor() as executor:
        compress_file_futures = []
        for file in files:
            file_no_ext = file.split(".")[0]
            extension = file.split(".")[1]

            if file_no_ext in orig_values or extension != "vasp":
                continue

            future_compress_file = executor.submit(compress_file_func, file, file_no_ext)
            compress_file_futures.append(future_compress_file)

        for future in as_completed(compress_file_futures):
            if write:
                file_no_ext, charge, mag, compress_duration, orig_fs, charge_fs, mag_fs = future.result()
                metrics[file_no_ext]["orig_file_size"] = orig_fs
                metrics[file_no_ext]["compressed_data_size"] = charge_fs + mag_fs
                # TODO: Add proper compression ratio metric with all files included
                orig_values[file_no_ext] = [charge, mag]
            else:
                file_no_ext, structure, charge, mag, _, dims, charge_compressed, mag_compressed, compress_duration = future.result()
                if len(charge_compressed) > 0 and len(mag_compressed) > 0:
                    compressed_values[file_no_ext] = [charge_compressed, mag_compressed, structure.lattice.matrix, dims]
                    orig_values[file_no_ext] = [charge, mag]
                else:
                    orig_values[file_no_ext] = [charge, mag, dims]

            metrics[file_no_ext]["compress_duration"] = compress_duration
            # TODO: Add file size metrics, mandate compression duration for both charge and mag

    if write:
        return orig_values, metrics
    else:
        return orig_values, compressed_values, metrics

def decompress_dir(files: list[str], decompress_file_func, compressor_name: str):
    decompressed_values = {}
    metrics = defaultdict(dict)
    with ThreadPoolExecutor() as executor:
        decompress_file_futures = []
        for file in files:
            file_no_ext = file.split(".")[0]
            future_decompress_file = executor.submit(decompress_file_func, file)
            decompress_file_futures.append(future_decompress_file)

        for future in as_completed(decompress_file_futures):
            if future.result():
                file_no_ext, structure, data_aug, charge, mag, decompress_duration = future.result()

                cgden = chgcar.remake_chgcar_pymatgen(
                    charge,
                    mag,
                    structure,
                    data_aug
                )
                cgden.write_file(f"{file_no_ext}_{compressor_name}.vasp")

                decompressed_values[file_no_ext] = [charge, mag]
                metrics[file_no_ext]["decompress_duration"] = decompress_duration

        return decompressed_values, metrics

def decompress_dir_no_file(compressed_values, decompress_func, has_data = True):
    decompressed_values = {}
    metrics = defaultdict(dict)
    with ThreadPoolExecutor() as executor:
        decompress_file_futures = []
        for file_no_ext, values in compressed_values.items():
            if has_data:
                charge_compressed, mag_compressed, lattice, dims = values
                future_decompress_file = executor.submit(decompress_func, file_no_ext, charge_compressed, mag_compressed, lattice, dims)
            else:
                _, _, dims = values
                future_decompress_file = executor.submit(decompress_func, file_no_ext, None, None, None, dims)
            decompress_file_futures.append(future_decompress_file)

        for future in as_completed(decompress_file_futures):
            if future.result():
                file_no_ext, charge, mag, decompress_duration = future.result()
                decompressed_values[file_no_ext] = [charge, mag]
                metrics[file_no_ext]["decompress_duration"] = decompress_duration

        return decompressed_values, metrics

# TODO: Implement
def remake_chgcar_dir(files: list[str], decompressed_values):

    with ThreadPoolExecutor() as executor:
        no_data_files = (fn for fn in files if "no_data" in fn)
        for file_name in no_data_files:
            file_name_prefix = file_name.split("_no_data")[0]

            try:
                charge_data, mag_data = decompressed_values[file_name_prefix]
                future_remake_chgcar = executor.submit(chgcar.remake_chgcar, file_name, charge_data, mag_data, file_name_prefix + "_final.vasp")

            except:
                print(f"Error: Could not find decompressed values for {file_name_prefix}")

def decompress_and_remake_dir(files: list[str], decompress_func):
    with ThreadPoolExecutor() as executor:
        future_decompress_values = executor.submit(decompress_dir, files, decompress_func)
        future_remake_chgcar_dir = executor.submit(remake_chgcar_dir, files, future_decompress_values.result())

# Helpers
def get_only_file_name(file: str):
    paths = file.split("/")
    only_file_name = paths[-1].split("_")[0] + "_chgcar"
    return "/".join(paths[:-1]) + "/" + only_file_name

def get_files_in_dir(directory: str):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def check_dir(directory: str):
    return os.path.exists(directory)

def check_files(files: list[str]):
    return all(os.path.exists(f) for f in files)

def delete_files(files: list[str]):
    for f in files:
        pathlib.Path(f).unlink()

def get_file_size_mb(file: str):
    return os.path.getsize(file) / (1024 * 1024)
