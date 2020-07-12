#!/usr/bin/env bash

set -e
unset PYTHONHOME
cython --embed -3 "$1"

PYTHON_LIBRARY=$(ls $CONDA_PREFIX/lib/libpython*.so| sed s/.so//g | egrep -o "python.{3,}")
gcc -march=native -I $CONDA_PREFIX/include/python* -L $CONDA_PREFIX/lib *.c -l$PYTHON_LIBRARY -o $2 -O3
echo "#/usr/bin/env bash" > run.sh
echo "export PYTHONHOME=$CONDA_PREFIX" >> run.sh
echo "export LD_LIBRARY_PATH=$LD_LIBRARYP_PATH:$CONDA_PREFIX/lib" >> run.sh
echo "" >> run.sh
echo "./$2 \$@" >> run.sh
chmod +x run.sh
