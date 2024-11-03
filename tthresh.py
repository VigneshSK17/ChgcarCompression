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
        orig_values, all_metrics = io2.compress_dir(files, compress_file_helper, "tthresh", write_raw=True)
        for file_no_ext, file_metrics in all_metrics.items():
            print(file_no_ext, "Compression Duration: ", file_metrics["compress_duration"], "s")

    if method == "decompress":
        decompressed_values = io.decompress_dir(files, decompress_func)
        for file_name, (charge, mag) in decompressed_values.items():
            print(file_name, charge.shape, mag.shape)

    if method == "remake":
        io.decompress_and_remake_dir(files, decompress_func)

def compress_file_helper(file: str, file_no_ext: str):
    structure, charge_pgrid, mag_pgrid, data_aug, dims = chgcar.parse_chgcar_pymatgen(file)

    charge = charge_pgrid.grid_data
    mag = mag_pgrid.grid_data

    chgcar.data_to_raw(charge, dims, f"{file_no_ext}_tthresh_charge.raw")
    chgcar.data_to_raw(mag, dims, f"{file_no_ext}_tthresh_mag.raw")

    charge_compress_duration = compress_func(file_no_ext, "charge", dims)
    mag_compress_duration = compress_func(file_no_ext, "mag", dims)

    chgcar.store_structure_aug_pymatgen(file_no_ext, structure, data_aug)

    return file_no_ext, charge, mag, charge_compress_duration + mag_compress_duration

def compress_func(chgcar_fn: str, section: str, dims: list[int]):
    time_start = perf_counter()
    cmd = get_tthresh_compress_cmd(chgcar_fn, section, dims)
    subprocess.run(cmd)
    time_end = perf_counter()

    return time_end - time_start

def get_file_no_ext(file: str):
    return file.split(".")[0]


def decompress_func(compressed_fn: str):
    start_decompress = resource.getrusage(resource.RUSAGE_CHILDREN)
    cmd = get_tthresh_decompress_cmd(compressed_fn)
    subprocess.run(cmd)
    end_decompress = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(f"{compressed_fn} Decompression Time: {end_decompress.ru_utime - start_decompress.ru_utime}s")

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