clean_compress:
	rm test/tthresh/*{.raw,no_data}
clean_decompress:
	rm test/tthresh_decompress/*decompressed.raw
clean_remake:
	rm test/tthresh_decompress/*{decompressed.raw,_final.vasp}

clean:
	-make clean_compress
	-make clean_decompress
	-make clean_remake