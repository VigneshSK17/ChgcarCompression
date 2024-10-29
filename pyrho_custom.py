from concurrent.futures import ThreadPoolExecutor, wait
import json
import sys
from time import perf_counter
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

from utils import chgcar, io

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
    np.savez_compressed(f"{chgcar_fn}_compressed.npz", charge=charge, mag=mag, dims=dims)

    with open(f"{chgcar_fn}_structure.cif", "w") as f:
        f.write(structure.to(fmt="cif"))
    open(f"{chgcar_fn}_data_aug.txt", "w").write(json.dumps(data_aug))

# TODO: Add methods to get compressed data, decompress, and remake CHGCAR
def retrieve_compressed(chgcar_fn: str):
    data = np.load(f"{chgcar_fn}_compressed.npz")
    charge_compressed = data["charge"]
    mag_compressed = data["mag"]
    dims = data["dims"]

    parser = CifParser(f"{chgcar_fn}_structure.cif")
    structure = parser.parse_structures()[0]
    lattice = structure.lattice.matrix
    data_aug = json.loads(open(f"{chgcar_fn}_data_aug.txt").read())

    return charge_compressed, mag_compressed, dims, structure, lattice, data_aug


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


def compress_dir(files: list[str]):
    orig_values = {}
    with ThreadPoolExecutor() as executor:
        for file in files:
            file_name = file.split(".")[0]
            extension = file.split(".")[1]

            if file_name in orig_values or extension != "vasp":
                continue

            future_parser = executor.submit(parse_chgcar, file)
            structure, charge, mag, data_aug, dims = future_parser.result()

            orig_values[file_name] = [charge, mag]

            future_compressed = executor.submit(compress_func, charge, mag, dims)

            charge_compressed, mag_compressed, compress_duration = future_compressed.result()

            future_store_compressed = executor.submit(store_compressed, file_name, charge_compressed, mag_compressed, structure, data_aug, dims)

            print(f"{file_name} - Compression Duration: {compress_duration} s")

    return orig_values

def decompress_and_remake_dir(files: list[str]):
    decompressed_values = {}
    with ThreadPoolExecutor() as executor:
        for file in files:
            paths = file.split("/")
            only_file_name = paths[-1].split("_")[0] + "_chgcar"
            file_name = "/".join(paths[:-1]) + "/" + only_file_name

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
        orig_values = compress_dir(files)
        for file_name, (charge, mag) in orig_values.items():
            print(file_name, charge.grid_shape, mag.grid_shape)

    if method == "decompress":
        decompressed_values = decompress_and_remake_dir(files)
        for file_name, (charge, mag) in decompressed_values.items():
            print(file_name, charge.grid_shape, mag.grid_shape)

    # Compresses and then decompresses CHGCAR's, provides MAE
    if method == "remake":
        orig_values = compress_dir(files)
        decompressed_values = decompress_and_remake_dir(files)

        for file_name in orig_values.keys():
            orig = orig_values[file_name]
            decompressed = decompressed_values[file_name]
            print(file_name, "Charge Density MAE: ", chgcar.mae(orig[0].grid_data, decompressed[0].grid_data))
            print(file_name, "Mag Density MAE: ", chgcar.mae(orig[1].grid_data, decompressed[1].grid_data))
            print(file_name, "Charge Density Avg Percentage Difference: ", chgcar.mean_percentage_diff(orig[0].grid_data, decompressed[0].grid_data))
            print(file_name, "Mag Density Avg Percentage Difference: ", chgcar.mean_percentage_diff(orig[1].grid_data, decompressed[1].grid_data))


if __name__ == "__main__":
    main()