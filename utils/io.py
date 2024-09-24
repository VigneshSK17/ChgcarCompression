import os
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from utils.chgcar import *

def compress_dir(files: list[str], compress_func, compressor_name: str, write_raw = False):

    orig_values = {}
    with ThreadPoolExecutor() as executor:
        for file in files:
            file_name = file.split(".")[0]

            future_parse = executor.submit(parse_chgcar, file, file_name + f"_{compressor_name}_no_data")

            future_raw_charge = executor.submit(
                data_to_raw,
                future_parse.result()[1], future_parse.result()[0],file_name + f"_{compressor_name}_charge.raw" if write_raw else None)
            future_raw_mag = executor.submit(
                data_to_raw, future_parse.result()[2], future_parse.result()[0],
                file_name + f"_{compressor_name}_mag.raw" if write_raw else None)

            # TODO: Prevent waiting for both compressions to be done
            wait([future_raw_charge, future_raw_mag])

            orig_values[file_name] = [future_raw_charge.result(), future_raw_mag.result()]

            future_compress_cmd_charge = executor.submit(compress_func, file_name, "charge", future_parse.result()[0])
            future_compress_cmd_mag = executor.submit(compress_func, file_name, "mag", future_parse.result()[0])

        return orig_values

def decompress_dir(files: list[str], decompress_func):
    decompressed_values = {}
    with ThreadPoolExecutor() as executor:
        for file in files:
            file_name = file.split(".")[0]

            if "no_data" in file_name:
                continue

            future_decompress = executor.submit(decompress_func, file_name)
            wait([future_decompress])

            future_decompress_values = executor.submit(raw_to_data, f"{file_name}_decompressed.raw")

            if "charge" in file_name:
                file_name_prefix = file_name.split("_charge")[0]

                if file_name_prefix not in decompressed_values:
                    decompressed_values[file_name_prefix] = [future_decompress_values.result()]
                else:
                    decompressed_values[file_name_prefix].insert(0, future_decompress_values.result())
            elif "mag" in file_name:
                file_name_prefix = file_name.split("_mag")[0]

                if file_name_prefix not in decompressed_values:
                    decompressed_values[file_name_prefix] = [future_decompress_values.result()]
                else:
                    decompressed_values[file_name_prefix].append(future_decompress_values.result())



            """
            TODO
            - Store values based on file_name_orig into dict
            - Combine all values into CHGCAR, get nodata files somehow (separate func?)
            """
        return decompressed_values

# TODO: Implement
def remake_chgcar_dir(files: list[str], decompressed_values):

    with ThreadPoolExecutor() as executor:
        no_data_files = (fn for fn in files if "no_data" in fn)
        for file_name in no_data_files:
            file_name_prefix = file_name.split("_no_data")[0]

            try:
                charge_data, mag_data = decompressed_values[file_name_prefix]
                future_remake_chgcar = executor.submit(remake_chgcar, file_name, charge_data, mag_data, file_name_prefix + "_final.vasp")

            except:
                print(f"Error: Could not find decompressed values for {file_name_prefix}")

def decompress_and_remake_dir(files: list[str], decompress_func):
    with ThreadPoolExecutor() as executor:
        future_decompress_values = executor.submit(decompress_dir, files, decompress_func)
        future_remake_chgcar_dir = executor.submit(remake_chgcar_dir, files, future_decompress_values.result())

# Helpers
def get_files_in_dir(directory: str):
    return [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

def check_dir(directory: str):
    return os.path.exists(directory)