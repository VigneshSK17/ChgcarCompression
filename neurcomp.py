import json
import numpy as np
import subprocess
import sys
from time import perf_counter

from pyrho.charge_density import PGrid
import vtk
from vtk.util import numpy_support


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
            "--volume", fn,
            "--network", network_fn,
            "--config", config_fn,
            # "--d_out", "3",
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


def decompress_func(compressed_fn: str, dims: list[int]):
    decompressed_fn = compressed_fn.split("_compressed")[0] + "_decompressed"

    # TODO: Figure out how to neurcomp decompress without orig volume
    cmd = ["python", NEURCOMP_DIR + "net_decompress.py",
        "--compressed", compressed_fn,
        "--resolution", f"{dims[0]}x{dims[1]}x{dims[2]}",
        "--recon", decompressed_fn]

    time_start = perf_counter()
    subprocess.run(cmd)
    time_end = perf_counter()

    return decompressed_fn + ".npy", time_end - time_start


def retrieve_compressed(file: str):
    if "_neurcomp_compressed" not in file:
        return None
    chgcar_fn = file.split("_")[0] + "_chgcar"
    files_required = [f"{chgcar_fn}_charge_neurcomp_compressed", f"{chgcar_fn}_mag_neurcomp_compressed", f"{chgcar_fn}_structure.cif", f"{chgcar_fn}_data_aug.txt"]
    # TODO: Prevent repeating reading all files for a particular VASP, only do it once
    if not io2.check_files(files_required):
        print(f"{file}: Missing files for decompression")
        return None

    structure, lattice, data_aug, dims = chgcar.retrieve_structure_aug_dims_pymatgen(chgcar_fn)

    return chgcar_fn, dims, structure, lattice, data_aug


def vti_to_array(*fns):
    arrays = []
    for fn in fns:
        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(fn)
        reader.Update()

        data = reader.GetOutput()
        point_data = data.GetPointData()
        scalar_array = point_data.GetScalars()

        numpy_array = numpy_support.vtk_to_numpy(scalar_array)
        dimensions = data.GetDimensions()
        arrays.append(numpy_array.reshape(dimensions[::-1]))

    return arrays


def compress_file_helper(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims, fs = chgcar.parse_chgcar_pymatgen(file)
    charge, mag = charge_pgrid.grid_data, mag_pgrid.grid_data

    charge_fn, mag_fn = file_no_ext + "_charge.npy", file_no_ext + "_mag.npy"
    np.save(file_no_ext + "_charge.npy", charge)
    np.save(file_no_ext + "_mag.npy", mag)

    charge_network_fn, charge_config_fn, charge_train_duration = train_func(charge_fn)
    mag_network_fn, mag_config_fn, mag_train_duration = train_func(mag_fn)

    charge_compressed_fn, charge_fs, charge_compress_duration = compress_func(charge_network_fn, charge_config_fn)
    mag_compressed_fn, mag_fs, mag_compress_duration = compress_func(mag_network_fn, mag_config_fn)

    chgcar.store_structure_aug_dims_pymatgen(file_no_ext, structure, data_aug, dims)

    total_train_duration = charge_train_duration + mag_train_duration
    total_compress_duration = charge_compress_duration + mag_compress_duration

    return file_no_ext, charge_pgrid, mag_pgrid, f"{total_train_duration},{total_compress_duration}", fs, charge_fs, mag_fs


def decompress_file_helper(file: str):
    retrieved = retrieve_compressed(file)
    if retrieved is None:
        return None
    chgcar_fn, dims, structure, lattice, data_aug = retrieved

    decompress_charge_fn, decompress_charge_duration = decompress_func(f"{chgcar_fn}_charge_neurcomp_compressed", dims)
    decompress_mag_fn, decompress_mag_duration = decompress_func(f"{chgcar_fn}_mag_neurcomp_compressed", dims)

    # charge_array, mag_array = vti_to_array(decompress_charge_fn, decompress_mag_fn)
    charge_array, mag_array = np.load(decompress_charge_fn), np.load(decompress_mag_fn)

    charge_pgrid, mag_pgrid = PGrid(charge_array, lattice), PGrid(mag_array, lattice)

    return chgcar_fn, structure, data_aug, charge_pgrid, mag_pgrid, decompress_charge_duration + decompress_mag_duration


def main():
    folder = sys.argv[1]
    method = sys.argv[2]

    if not io2.check_dir(folder):
        print("Invalid directory")
        sys.exit(1)
    files = io2.get_files_in_dir(folder)

    if method == "compress":
        orig_values, all_metrics = io2.compress_dir(files, compress_file_helper, "neurcomp")
        print(json.dumps(all_metrics, sort_keys=True, indent=4))

    elif method == "decompress":
        decompressed_values, all_metrics = io2.decompress_dir(files, decompress_file_helper, "neurcomp")
        print(json.dumps(all_metrics, sort_keys=True, indent=4))

    elif method == "remake":
        print("Starting compression...")
        orig_values, compress_metrics = io2.compress_dir(files, compress_file_helper, "neurcomp")

        print("Starting decompression...")
        files = io2.get_files_in_dir(folder) # Updates files
        decompressed_values, decompress_metrics = io2.decompress_dir(files, decompress_file_helper, "neurcomp")

        all_metrics = chgcar.generate_metrics(orig_values, decompressed_values, compress_metrics, decompress_metrics)
        print(json.dumps(all_metrics, sort_keys=True, indent=4))


if __name__ == "__main__":
    main()
