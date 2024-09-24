import numpy as np

import math


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


# Math
def mae(actual: np.ndarray, predicted: np.ndarray):
    return np.average(np.absolute((actual.astype("float") - predicted.astype("float"))))