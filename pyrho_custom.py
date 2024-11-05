from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import sys
from time import perf_counter
import numpy as np
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

def parse_chgcar(chgcar_fn: str):
    vasp_cden = Chgcar.from_file(chgcar_fn)
    cden = ChargeDensity.from_file(chgcar_fn)

    structure: Structure = cden.structure
    charge = cden.pgrids["total"]
    mag = cden.pgrids["diff"]
    data_aug = vasp_cden.as_dict()["data_aug"]
    dims = cden.grid_shape

    return structure, charge, mag, data_aug, dims

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

    chgcar.store_structure_aug_dims_pymatgen(chgcar_fn, structure, data_aug, dims)

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


def compress_file_helper(file: str, file_no_ext: str):
    structure, charge, mag, data_aug, dims = parse_chgcar(file)
    charge_compressed, mag_compressed, compress_duration = compress_func(charge, mag, dims)
    store_compressed(file_no_ext, charge_compressed, mag_compressed, structure, data_aug, dims)

    return file_no_ext, charge, mag, compress_duration

def decompress_file_helper(file: str):
    if retrieve_compressed(file) is None:
        return None
    chgcar_fn, charge_compressed, mag_compressed, dims, structure, lattice, data_aug = retrieve_compressed(file)

    decompress_charge, decompress_charge_duration = decompress_func(charge_compressed, lattice, dims)
    decompress_mag, decompress_mag_duration = decompress_func(mag_compressed, lattice, dims)

    return chgcar_fn, structure, data_aug, decompress_charge, decompress_mag, decompress_charge_duration + decompress_mag_duration

def compress_dir(files: list[str]):
    orig_values = {}
    metrics = defaultdict(dict)

    with ThreadPoolExecutor() as executor:
        compress_file_futures = []
        for file in files:
            file_no_ext = file.split(".")[0]
            extension = file.split(".")[1]

            if file_no_ext in orig_values or extension != "vasp":
                continue

            future_compress_file = executor.submit(compress_file_helper, file, file_no_ext)
            compress_file_futures.append(future_compress_file)

        for future in as_completed(compress_file_futures):
            file_no_ext, charge, mag, compress_duration = future.result()
            orig_values[file_no_ext] = [charge, mag]
            metrics[file_no_ext]["compress_duration"] = compress_duration
            # TODO: Add file size metrics

    return orig_values, metrics

def decompress_and_remake_dir(files: list[str]):
    decompressed_values = {}
    with ThreadPoolExecutor() as executor:
        for file in files:

            if file_name in decompressed_values:
                continue

            future_decompress = executor.submit(retrieve_compressed, file_name)
            charge_compressed, mag_compressed, dims, structure, lattice, data_aug = future_decompress.result()

            # shape = [dim * int(sys.argv[3]) for dim in charge_compressed.shape]

            # future_decompress_charge = executor.submit(decompress_func, charge_compressed, lattice, shape)
            # future_decompress_mag = executor.submit(decompress_func, mag_compressed, lattice, shape)
            future_decompress_charge = executor.submit(decompress_func, charge_compressed, lattice, dims)
            future_decompress_mag = executor.submit(decompress_func, mag_compressed, lattice, dims)


            decompress_charge, decompress_charge_duration = future_decompress_charge.result()
            decompress_mag, decompress_mag_duration = future_decompress_mag.result()

            decompressed_values[file_name] = [decompress_charge, decompress_mag]

            # Remaking chgcar
            future_remake_chgcar = executor.submit(remake_chgcar, file_name, decompress_charge, decompress_mag, structure, data_aug)

            future_remake_chgcar.result()

            print(f"{file_name} - Decompression Duration: {decompress_charge_duration + decompress_mag_duration} s\n\t Charge Decompression: {decompress_charge_duration} s\n\t Mag Decompression: {decompress_mag_duration} s")

    return decompressed_values



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

        all_metrics = defaultdict(dict)
        for file_no_ext in compress_metrics.keys():
            for k, v in compress_metrics[file_no_ext].items():
                all_metrics[file_no_ext][k] = v
            for k, v in decompress_metrics[file_no_ext].items():
                all_metrics[file_no_ext][k] = v

        for file_no_ext in orig_values.keys():
            orig, decompressed = orig_values[file_no_ext], decompressed_values[file_no_ext]
            all_metrics[file_no_ext]["charge_mae"] = chgcar.mae(orig[0].grid_data, decompressed[0].grid_data)
            all_metrics[file_no_ext]["mag_mae"] = chgcar.mae(orig[1].grid_data, decompressed[1].grid_data)
            all_metrics[file_no_ext]["charge_avg_percentage_diff"] = chgcar.mean_percentage_diff(orig[0].grid_data, decompressed[0].grid_data)
            all_metrics[file_no_ext]["mag_avg_percentage_diff"] = chgcar.mean_percentage_diff(orig[1].grid_data, decompressed[1].grid_data)

        print(json.dumps(all_metrics, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()