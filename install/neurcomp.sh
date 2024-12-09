#!/bin/sh

# Get parent directory of this script
PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
cd $PROJECT_DIR/..
pwd

# Make lib directory
LIB_DIR="$PROJECT_DIR/lib"
if [ ! -d "$LIB_DIR" ]; then
		mkdir "$LIB_DIR"
fi
# Check if tthresh exists
NEURCOMP_DIR="$LIB_DIR/neurcomp"
if [ -d "$NEURCOMP_DIR" ]; then
	echo "neurcomp directory already exists, delete for reinstall"
	exit 0
else
	mkdir $NEURCOMP_DIR
fi

cd $LIB_DIR
git clone https://github.com/matthewberger/neurcomp.git

cd $LIB_DIR/neurcomp
sed 's/==.*$//' requirements.txt | xargs pip install