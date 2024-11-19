#!/bin/sh

# Get parent directory of this script
PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd $PROJECT_DIR/..
pwd

# Make lib directory
LIB_DIR="./lib"
if [ ! -d "$LIB_DIR" ]; then
		mkdir "$LIB_DIR"
fi
# Check if tthresh exists
SZ3_DIR="$LIB_DIR/sz3"
if [ -d "$SZ3_DIR" ]; then
	echo "sz3 binary directory already exists, delete for reinstall"
	exit 0
else
	mkdir $SZ3_DIR
fi
# SZ3_DIR="$LIB_DIR/sz3/bin" # TODO: Fix this

cd $LIB_DIR
git clone https://github.com/szcompressor/SZ3.git
cd SZ3 && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX:PATH="$SZ3_DIR" ..
make
make install