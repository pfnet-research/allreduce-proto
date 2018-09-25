from subprocess import Popen
from typing import List
from typing import Optional
from typing import Union
import itertools
import os
import os.path
import pytest
import unittest
import sys


IBCOMM_INVALID_ARGUMENT = 1
IBCOMM_IBVERBS_ERROR = 2
IBCOMM_CUDA_ERROR = 3
IBCOMM_NOT_SUPPORTED = 4

ALGO_RING = "ring"
ALGO_RABEN = "rabenseifner"

IBCOMM_ALGORITHMS = [ALGO_RING, ALGO_RABEN]


def find_file(directory: str, fname: str) -> Optional[str]:
    """Find a file in directory. Used to find the allreduce_tester binary."""
    for root, dirs, files in os.walk(directory):
        if fname in files:
            return os.path.join(root, fname)

    # not found
    return None


def flatten1(lst: List[List]) -> List:
    return [item for sublist in lst for item in sublist]


def dict_to_envs(envs: dict) -> List[str]:
    """
    Expand a dict to command line arguments for Open MPI's '-x' option
    e.g.) {'FOO': 100} 
           ==> 
          ['-x', 'FOO=100']
    """
    z = itertools.zip_longest([],
                              ["{}={}".format(k,v) for k, v in envs.items()],
                              fillvalue='-x')

    return flatten1(list(z))


# Find the project directory
ProjectDir = os.path.join(os.path.dirname(__file__), os.pardir)
Tester = find_file(ProjectDir, 'allreduce_tester')

if not os.path.exists(Tester):
    sys.stderr.write("Please build 'allreduce_tester' before running unit tests.\n")
    exit(1)


class AllreduceTest(unittest.TestCase):
    @staticmethod
    def check(np: Union[int, str],
              algo: str,
              buffsize: Union[int, str],
              init_expr: str = "i*np+p",
              check_expr: str = "i*np*np+np*(np-1)/2",
              chunksize: Optional[Union[str, int]] = None):
        env = {}
        if chunksize is not None:
            env['IBCOMM_CHUNKSIZE'] = chunksize

        if 'NODEFILE' in os.environ:
            hostfile = os.environ['NODEFILE']
        elif 'PBS_NODEFILE' in os.environ:
            hostfile = os.environ['PBS_NODEFILE']
        else:
            hostfile = None

        assert algo in IBCOMM_ALGORITHMS, "{} is not support Allreduce algorithm.".format(algo)

        env_args = dict_to_envs(env)
        np = str(np)
        buffsize = str(buffsize)

        if hostfile is not None:
            hostfile = ['--hostfile', hostfile]
        else:
            hostfile = []

        cmd = ['timeout', '90s', 'mpiexec', '-np', np, *hostfile, *env_args, Tester, algo, buffsize, init_expr, check_expr]

        print()
        print(' '.join(cmd))
        p = Popen(cmd)
        out,err = p.communicate()
        return p.returncode

    def setUp(self):
        pass

    def test_1proc(self):
        # Allreduce works with just 1 process.
        ret = AllreduceTest.check(algo=ALGO_RING, np=1, buffsize=1024)
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RABEN, np=1, buffsize=1024)
        assert ret == 0

    def test_small_buffer(self):
        # Tests Ring-AllReduce
        int_size = 4

        # Allreduce works with small buffer size
        NP=1
        ret = AllreduceTest.check(algo=ALGO_RING, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=2
        ret = AllreduceTest.check(algo=ALGO_RING, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=3
        ret = AllreduceTest.check(algo=ALGO_RING, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=5
        ret = AllreduceTest.check(algo=ALGO_RING, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=2
        # Relatively larger prime
        ret = AllreduceTest.check(algo=ALGO_RING, np=NP, buffsize=2521 * int_size)
        assert ret == 0

        # Tests Rabenseifner's algorithm
        NP=1
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=2
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=NP, buffsize=NP * 2 * int_size)
        assert ret == 0

        NP=3
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=NP, buffsize=NP * 2 * int_size)
        assert ret == IBCOMM_NOT_SUPPORTED  # Currently, non-power-of-2 np is not supported.

        NP=5
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=NP, buffsize=NP * 2 * int_size)
        assert ret == IBCOMM_NOT_SUPPORTED  # Currently, non-power-of-2 np is not supported.

        NP=2
        # Relatively larger prime
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=NP, buffsize=2521 * int_size)
        assert ret == 0

    def test_basic(self):
        # Relatively larger buffer size and default chunksize
        # Tests Ring-AllReduce
        ret = AllreduceTest.check(algo=ALGO_RING, np=2, buffsize="128M")
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=3, buffsize="128M")
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="128M")
        assert ret == 0

        # Test Rabenseifner's algorithm
        ret = AllreduceTest.check(algo=ALGO_RABEN, np=2, buffsize="128M")
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RABEN, np=3, buffsize="128M")
        assert ret == IBCOMM_NOT_SUPPORTED  # Currently, non-power-of-2 np is not supported.

        ret = AllreduceTest.check(algo=ALGO_RABEN, np=4, buffsize="128M")
        assert ret == 0

    def test_chunk_size(self):
        # Tests Ring-AllReduce
        # for a buffer size 1024 and NP 2, change the IBCOMM_CHUNKSIZE from [4 to 128]
        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='4')
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='8')
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='16')
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='32')
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='64')
        assert ret == 0

        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize='128')
        assert ret == 0

        # Test of Rabenseifner's algorithm is not necessary because chunksize is not used.

    def test_invalid_error(self):
        # Test if ibcomm checks chunk size
        int_size = 4
        for chunk_size in range(0, int_size):  # try 0, 1, 2, 3
            # chunk_size must be a multiply of element type (which is int here)
            ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize=chunk_size)
            assert ret == IBCOMM_INVALID_ARGUMENT

        # Chunk size < 0
        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize="-128")
        assert ret == IBCOMM_INVALID_ARGUMENT

        # Chunk size is too large
        ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="1k", chunksize="1k")
        assert ret == IBCOMM_INVALID_ARGUMENT

        # Check too short vector
        int_size = 4
        ret = AllreduceTest.check(algo=ALGO_RING, np=2, buffsize=int_size, init_expr="1", check_expr="np")
        assert ret == IBCOMM_NOT_SUPPORTED

        ret = AllreduceTest.check(algo=ALGO_RABEN, np=2, buffsize=int_size, init_expr="1", check_expr="np")
        assert ret == IBCOMM_NOT_SUPPORTED

    @pytest.mark.slow
    def test_aging(self):
        for i in range(100):
            ret = AllreduceTest.check(algo=ALGO_RING, np=4, buffsize="128M")
            assert ret == 0

        for i in range(100):
            ret = AllreduceTest.check(algo=ALGO_RABEN, np=4, buffsize="128M")
            assert ret == 0
