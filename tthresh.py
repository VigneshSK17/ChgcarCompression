import resource
import sys
import subprocess

from utils import io

TTHRESH_BIN = "./lib/tthresh/build/tthresh"

def main():
    folder = sys.argv[1]
    method = sys.argv[2]
    # error_bound, error_value = sys.argv[3], sys.argv[4]

    if not io.check_dir(folder):
        print("Invalid directory")
        sys.exit(1)

    files = io.get_files_in_dir(folder)

    if method == "compress":
        compressed_values = io.compress_dir(files, compress_func, "tthresh", write_raw=True)
        for file_name, (charge, mag) in compressed_values.items():
            print(file_name, charge.shape, mag.shape)

    if method == "decompress":
        decompressed_values = io.decompress_dir(files, decompress_func)
        for file_name, (charge, mag) in decompressed_values.items():
            print(file_name, charge.shape, mag.shape)

    if method == "remake":
        io.decompress_and_remake_dir(files, decompress_func)



def compress_func(chgcar_fn: str, section: str, dims: list[int]):
    start_compress = resource.getrusage(resource.RUSAGE_CHILDREN)
    cmd = get_tthresh_compress_cmd(chgcar_fn, section, dims)
    subprocess.run(cmd)
    end_compress = resource.getrusage(resource.RUSAGE_CHILDREN)

    print(f"{chgcar_fn} Compression Time: {end_compress.ru_utime - start_compress.ru_utime}s")

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