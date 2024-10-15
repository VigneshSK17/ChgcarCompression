from concurrent.futures import ThreadPoolExecutor, wait
import json
import sys
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

from utils import io

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
    lattice = cden.lattice
    data_aug = vasp_cden.as_dict()["data_aug"]
    dims = cden.grid_shape

    return structure, charge, mag, lattice, data_aug, dims

# TODO: Time using timeit.timeit not resource.getrusage
def compress_func(charge: PGrid, mag: PGrid, dims: list[int]):
    dims = charge.grid_shape
    compressed_dims = [dim // int(sys.argv[3]) for dim in dims]

    charge_compressed = charge.lossy_smooth_compression(compressed_dims, float(sys.argv[4]))
    mag_compressed = mag.lossy_smooth_compression(compressed_dims, float(sys.argv[4]))

    return charge_compressed, mag_compressed


def store_compressed(chgcar_fn: str, charge, mag, structure, lattice, data_aug):
    np.savez_compressed(f"{chgcar_fn}_compressed.npz", charge=charge, mag=mag)

    with open(f"{chgcar_fn}_structure.cif", "w") as f:
        f.write(structure.to(fmt="cif"))
    np.savez_compressed(f"{chgcar_fn}_lattice.npz", lattice=lattice)
    open(f"{chgcar_fn}_data_aug.txt", "w").write(json.dumps(data_aug))

# TODO: Add methods to get compressed data, decompress, and remake CHGCAR
def retrieve_compressed(chgcar_fn: str):
    data = np.load(f"{chgcar_fn}_compressed.npz")
    charge_compressed = data["charge"]
    mag_compressed = data["mag"]

    parser = CifParser(f"{chgcar_fn}_structure.cif")
    structure = parser.parse_structures()[0]
    lattice = np.load(f"{chgcar_fn}_lattice.npz")["lattice"]
    data_aug = json.loads(open(f"{chgcar_fn}_data_aug.txt").read())

    return charge_compressed, mag_compressed, structure, lattice, data_aug


def decompress_func(data: np.ndarray, lattice: np.ndarray, dims: list[int]):
    pgrid = PGrid(data, lattice)
    up_sample_ratio = dims[0] // data.shape[0]
    pgrid_upscale = pgrid.get_transformed(
        sc_mat=np.eye(3),
        grid_out=dims,
        up_sample=up_sample_ratio
        # up_sample=1
    )
    return pgrid_upscale

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

            future_parser = executor.submit(parse_chgcar, file)
            structure, charge, mag, lattice, data_aug, dims = future_parser.result()

            orig_values[file_name] = [charge, mag]

            future_compressed = executor.submit(compress_func, charge, mag, dims)

            charge_compressed, mag_compressed = future_compressed.result()

            future_store_compressed = executor.submit(store_compressed, file_name, charge_compressed, mag_compressed, structure, lattice, data_aug)

    return orig_values

# TODO: Add multithreaded methods to decompress & remake CHGCAR

def main():
    folder = sys.argv[1]
    method = sys.argv[2]

    if not io.check_dir(folder):
        print("Invalid directory")
        sys.exit(1)
    files = io.get_files_in_dir(folder)

    if method == "compress":
        compressed_values = compress_dir(files)
        for file_name, (charge, mag) in compressed_values.items():
            print(file_name, charge, mag)

    if method == "decompress":
        pass

    if method == "remake":
        pass


if __name__ == "__main__":
    main()