#!/bin/sh

# TODO: Make sure this works for Linux & Mac

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
TTHRESH_DIR="$LIB_DIR/tthresh"
if [ -d "$TTHRESH_DIR" ]; then
	echo "tthresh binary directory already exists, delete for reinstall"	
	exit 0
else
	mkdir $TTHRESH_DIR
fi
TTHRESH_DIR="../../../bin/tthresh"

# Compile
cd $LIB_DIR
git clone https://github.com/rballester/tthresh.git
mkdir tthresh/build
cd tthresh/build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$TTHRESH_DIR" ..
make

# TODO: source tthresh in a separate script which combines all scripts
# 
