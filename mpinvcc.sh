#!/bin/bash

# Dirty hack :)
if echo "$*" | grep -qE -- '-DUSE_CUDA' ; then
    NVCC=1
fi

if [ "$NVCC" == 1 ]; then
    export OMPI_CXX=nvcc
    export MPICH_CXX=nvcc
    nvcc_arch_flag="-arch sm_52"

    if echo "$*" | grep -qE "(.cpp|.cxx|.cc)$" ; then
        xflag="-x cu"
    else
        xflag=
    fi
else
    nvcc_arch_flag=
    if [ -n "$CXX" ]; then
        export OMPI_CXX=${CXX}
        export MPICH_CXX=${CXX}
    fi
fi

CMD=$(mpicxx -show "$@" "$xflag" "$nvcc_arch_flag")

if [ "$NVCC" == 1 ]; then
    CMD=$(echo $CMD | sed -e "s/-Wl,/-Xlinker /g")
    CMD=$(echo $CMD | sed -e "s/\(-W[^ ][^ ]*\)/-Xcompiler \\1/g")
    CMD=$(echo $CMD | sed -e "s/\\(-pthread\\)/-Xcompiler \\1/g")
    CMD=$(echo $CMD | sed -e "s/\\(-rdynamic\\)/-Xcompiler \\1/g")
    CMD=$(echo $CMD | sed -e "s/\\(-fPIC\\)/-Xcompiler \\1/g")
fi

echo $CMD
$CMD
