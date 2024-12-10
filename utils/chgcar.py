from collections import defaultdict
import json
import numpy as np
from pymatgen.core.structure import Structure

from pymatgen.io.cif import CifParser
from pymatgen.io.vasp.outputs import Chgcar
from pyrho.charge_density import ChargeDensity, PGrid

import math

from utils import io2


# CHGCAR Stuff
def parse_chgcar(chgcar_fn, no_data_fn):

    charge, mag = [], []
    dims = None
    is_charge, is_mag = False, False

    with open(chgcar_fn, "r") as fi:

        lines = fi.read().splitlines()
        fo = open(no_data_fn, "w")

        i = 0
        while i < len(lines):
            if not is_charge and not is_mag:
                fo.write(lines[i] + "\n")

            if not charge and not lines[i].strip():
                dims = lines[i + 1]
                is_charge = True
                fo.write(lines[i + 1] + "\n")
                i += 1
            elif is_charge and "augmentation" not in lines[i]:
                line_nums = lines[i].strip().split(" ")
                for num in line_nums:
                    charge.append(float(num))
            elif is_charge and "augmentation" in lines[i]:
                fo.write(lines[i] + "\n")
                is_charge = False
            elif not mag and dims == lines[i]:
                is_mag = True
            elif is_mag and "augmentation" not in lines[i]:
                line_nums = lines[i].strip().split(" ")
                for num in line_nums:
                    mag.append(float(num))
            elif is_mag and "augmentation" in lines[i]:
                fo.write(lines[i] + "\n")
                is_mag = False

            i += 1

        fo.close()

    dims = [int(dim) for dim in dims.strip().split()]

    return dims, charge, mag


def remake_chgcar(no_data_fn: str, charge: np.ndarray, mag: np.ndarray, output_fn: str):

    fi = open(no_data_fn, "r")
    fo = open(output_fn, "w")

    fi_lines = fi.read().splitlines()
    fi.close()
    is_charge, is_mag, is_charge_done, dims = False, False, False, None
    i = 0

    charge = charge.reshape(-1)
    mag = mag.reshape(-1)

    while i < len(fi_lines):
        if not is_charge and not is_mag:
            fo.write(fi_lines[i] + "\n")
        if not dims and not fi_lines[i].strip():
            dims = fi_lines[i + 1]
            is_charge = True
            fo.write(fi_lines[i + 1] + "\n")
            i += 1
            write_data(charge, fo)
        elif is_charge and "augmentation" in fi_lines[i]:
            fo.write(fi_lines[i] + "\n")
            is_charge = False
            is_charge_done = True
        elif not is_mag and is_charge_done and fi_lines[i] == dims:
            is_mag = True
            write_data(mag, fo)
        elif is_mag and "augmentation" in fi_lines[i]:
            fo.write(fi_lines[i] + "\n")
            is_mag = False

        i += 1


    fo.close()


def write_data(data: np.ndarray, fo):
    line = ""
    for i, num in enumerate(data):
        if i % 5 == 0 and i > 0:
            fo.write(f" {line[:-1]}\n")
            line = ""

        if num == 0:
            base = 0
            exponent = 0
        else:
            exponent = math.floor(math.log10(abs(num)))
            base = num / (10 ** exponent)
            base /= 10
            exponent += 1


        if base < 0:
            base_str = f"{base:.11f}"
            line += base_str[0]
            line += base_str[2:]
        else:
            line += f"{base:.11f}"

        if exponent < 0:
            line += f"E-{-exponent:02d} "
        else:
            line += f"E+{exponent:02d} "

    fo.write(f" {line[:-1]}\n")


# Data Files
def data_to_raw(data: list[int], dims: list[int], output_file=None):
    arr_1d = np.array(data)

    if output_file:
        with open(output_file, "wb") as fo:
            arr_3d = arr_1d.reshape(dims)
            fo.write(arr_3d.tobytes())

    # TODO: Modify if 3D data needed
    return arr_1d


def raw_to_data(raw_file: str):
    with open(raw_file, "rb") as fi:
        arr = np.frombuffer(fi.read())
        return arr

# Pymatgen Methods
def parse_chgcar_pymatgen(chgcar_fn: str):
    vasp_cden = Chgcar.from_file(chgcar_fn)
    cden = ChargeDensity.from_file(chgcar_fn)

    structure: Structure = cden.structure
    charge = cden.pgrids["total"]
    mag = cden.pgrids["diff"]
    data_aug = vasp_cden.as_dict()["data_aug"]
    dims = cden.grid_shape

    fs = io2.get_file_size_mb(chgcar_fn)

    return structure, charge, mag, data_aug, dims, fs


def store_structure_aug_dims_pymatgen(file_no_ext: str, structure: Structure, data_aug, dims: list[int]):
    with open(f"{file_no_ext}_structure.cif", "w") as f:
        f.write(structure.to(fmt="cif"))
    open(f"{file_no_ext}_data_aug.txt", "w").write(json.dumps(data_aug))
    open(f"{file_no_ext}_dims.txt", "w").write(json.dumps(dims))

def retrieve_structure_aug_dims_pymatgen(file_no_ext: str):
    parser = CifParser(f"{file_no_ext}_structure.cif")
    structure = parser.parse_structures()[0]
    lattice = structure.lattice.matrix
    data_aug = json.loads(open(f"{file_no_ext}_data_aug.txt").read())
    with open(f"{file_no_ext}_dims.txt", "r") as fd:
        dims = json.load(fd)

    return structure, lattice, data_aug, dims

def remake_chgcar_pymatgen(charge_pgrid: PGrid, mag_pgrid: PGrid, structure: Structure, data_aug):
    cgden = ChargeDensity(pgrids={"total": charge_pgrid, "diff": mag_pgrid}, structure=structure)

    chgcar = cgden.to_Chgcar()
    chgcar.data_aug = data_aug
    return chgcar

def generate_metrics(orig_data, decompressed_data, compress_metrics, decompress_metrics):
    all_metrics = defaultdict(dict)
    for file_no_ext in compress_metrics.keys():
        for k, v in compress_metrics[file_no_ext].items():
            all_metrics[file_no_ext][k] = v
        for k, v in decompress_metrics[file_no_ext].items():
            all_metrics[file_no_ext][k] = v

    for file_no_ext in orig_data.keys():
        orig, decompressed = orig_data[file_no_ext], decompressed_data[file_no_ext]
        all_metrics[file_no_ext]["charge_mae"] = mae(orig[0].grid_data, decompressed[0].grid_data)
        all_metrics[file_no_ext]["mag_mae"] = mae(orig[1].grid_data, decompressed[1].grid_data)
        all_metrics[file_no_ext]["charge_avg_percentage_diff"] = mean_percentage_diff(orig[0].grid_data, decompressed[0].grid_data)
        all_metrics[file_no_ext]["mag_avg_percentage_diff"] = mean_percentage_diff(orig[1].grid_data, decompressed[1].grid_data)

    return all_metrics

def write_metrics_to_file(output_file, new_metrics, entry_name):
    with open(output_file, "r") as f:
        try:
            metrics_file_json = json.load(f)
        except json.JSONDecodeError:
            metrics_file_json = {}
    metrics_file_json[entry_name] = new_metrics
    with open(output_file, "w") as f:
        json.dump(metrics_file_json, f, sort_keys=True, indent=4)

# Math
def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.average(np.absolute((actual.astype("float") - predicted.astype("float"))))

def mean_percentage_diff(actual: np.ndarray, predicted: np.ndarray):
    actual, predicted = actual.astype("float"), predicted.astype("float")
    return np.sum(np.abs(actual - predicted))/np.sum(np.abs(actual))*100

# Random
def gen_df(values, name="orig"):
    values_dict = {}
    for key in values:
        values_dict[f'{key}_charge_{name}'] = values[key][0].grid_data.flatten()
        values_dict[f'{key}_mag_{name}'] = values[key][1].grid_data.flatten()

    return pd.DataFrame(dict([(key, pd.Series(value)) for key, value in values_dict.items()]))













