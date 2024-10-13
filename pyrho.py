import json
import sys
import numpy as np
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

def parse_chgcar(chgcar_fn: str):
    vasp_cden = Chgcar.from_file(chgcar_fn)
    cden = ChargeDensity.from_file(chgcar_fn)

    structure: Structure = cden.structure
    charge = cden.pgrids["total"]
    mag = cden.pgrids["diff"]
    lattice = cden.lattice
    data_aug = vasp_cden.as_dict()["data_aug"]

    return structure, charge, mag, lattice, data_aug

# TODO: Time using timeit.timeit not resource.getrusage
def compress_func(charge: PGrid, mag: PGrid, dims: list[int]):
    # TODO: sys.argv[2] = dims_divisor, sys.argv[3] = smear_std
    compressed_dims = [dim // sys.argv[2] for dim in dims]

    charge_compressed = charge.lossy_smooth_compression(dims, sys.argv[3])
    mag_compressed = mag.lossy_smooth_compression(dims, sys.argv[3])

    return charge_compressed, mag_compressed


def store_compressed(chgcar_fn: str, charge, mag, structure, lattice, data_aug):
    np.savez_compressed(f"{chgcar_fn}_compressed.npz", charge=charge, mag=mag)

    with open(f"{chgcar_fn}_structure.cif", "w") as f:
        f.write(structure.to(fmt="cif"))
    np.savez_compressed(f"{chgcar_fn}_lattice.npz", lattice=lattice)
    open(f"{chgcar_fn}_data_aug.txt", "w").write(json.dumps(data_aug))

# TODO: Add methods to get compressed data, decompress, and remake CHGCAR