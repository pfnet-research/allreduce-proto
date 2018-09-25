# PFNProto: AllReduce prototype implementation for NVIDIA GPUs and InfiniBand
PFNProto is a prototype library of the AllReduce collective operation.
The library is highly optimized for widely-used deep learning clusters equipped with NVIDIA GPUs and InfiniBand interconnect. It can achieve competitive performance to the fastest libraries in the world including NVIDIA NCCL.

PFNProto implements the following algorithms.
- Ring-AllReduce for CPU / CUDA
- Rabenseifner's Algorithm for CPU / CUDA

For more details, please refer to our blog ([English](https://preferredresearch.jp/2018/07/10/technologies-behind-distributed-deep-learning-allreduce/), [Japanese](https://research.preferred.jp/2018/07/prototype-allreduce-library/))

# How to build
## Dependencies
- Infiniband Verbs
- CMake 2.8+
- MPI (to build examples and tests)

## Build
```sh
mkdir build
cd build
cmake ..
make
```

# How to try the example
- Requirements: multi-node computing cluster equipped with NVIDIA GPUs and InfiniBand.
- Prepare hostfile
- Execute `examples/allreduce_cuda`
  - When PPN is not 1, this implementation isn't optimized. Therefore, selecting PPN=1 is needed.
  - (for Open MPI users) : `mpiexec -N 1 --hostfile "path_to_hostfile" examples/allreduce_cuda`
  - (for MPICH / MVAPICH2 users) : `mpiexec --ppn 1 --hostfile "path_to_hostfile" examples/allreduce_cuda`
- You'll see below result
```
$ cd build
$ (prepare your hostfile)
$ cat hosts
node01
node02
...
node08
$ mpiexec -N 1 --hostfile hosts examples/allreduce_cuda
rank: 3 OK
rank: 2 OK
rank: 4 OK
rank: 5 OK
rank: 6 OK
rank: 7 OK
rank: 1 OK
rank: 0 OK
elapsed time : 9.750750e-02 [s]
```

# Contribution
Any contributions to this prototype are welcome.
Please feel free to report an issue or send a pull request!

# Limitations
This library is a prototype for algorithm and performance demonstration purpose.
It is not intended to be used in a production environment.

In particular, there are several limitations including:
 - Python binding is not provided.
 - Not being designed to be used together with ChainerMN.
 - Supported reduction operation is only plus(+).
 - Non-power-of-two extension of Rabenseifner's algorithm is not implemented.
 - It currently focuses on inter-node communication. Intra-node communication is not efficient because shared memory or GPU-to-GPU DMA data transfer is not implemented.

# Tuning Knobs
You can control runtime behaviours of PFNProto through the following environment variables.
Memory or buffer size are all in [byte] and SI prefix is not supported.

## IBCOMM_CHUNKSIZE
- Chunk size in bytes for the AllReduce algorithm (both Ring-AllReduce and Rabenseifner's). PFNProto uses this size as a unit to execute every pipeline operation such as send, recv and reduction.
- Default value: (len(send/recvbuf) in bytes) / (4 * N_OF_PROCESSES)
- Supported range: (IBCOMM_CHUNKSIZE) <= (len(send/recvbuf) in bytes),  / (2 * N_OF_PROCESSES).

## IBCOMM_MEMORY_POOL_PRE_ALLOC
- PFNProto allocates this size of MemoryPool to hide the latency of memory allocation.
- Default value: 67108864 (64 MiB).

## IBCOMM_WORK_GPU_MEMORY_SIZE
- Initial Working GPU Memory size. PFNProto needs Working GPU Memory in rabenseifner's algorithm to save a reduction result. If this size is smaller than `IBCOMM_CHUNKSIZE`, runtime memory reallocation occurs.
- Default value: 33554432 (32 MiB).

## IBCOMM_NUM_CUDA_STREAM
- Total number of CUDA streams used.
- Default value: 64

## Number of pre-allocated chunks
- Number of pre-allocated chunks is computed from `IBCOMM_CHUNKSIZE` and `IBCOMM_MEMORY_POOL_PRE_ALLOC` by following equation:
- (IBCOMM_MEMORY_POOL_PRE_ALLOC) / (IBCOMM_CHUNKSIZE)

- Example: Let IBCOMM_CHUNKSIZE be 4 [MiB], and IBCOMM_MEMORY_POOL_PRE_ALLOC be 64 [MiB]. (IBCOMM_MEMORY_POOL_PRE_ALLOC) / (IBCOMM_CHUNKSIZE) = 64 / 4 = 16. Therefore, 16 chunks will be allocated and size of chunk is 4 [MiB].

# APIs
```c++
class IBVerbsCommunicator {
 public:
  explicit IBVerbsCommunicator(int world_size);

  // Manages infiniband-related resources thus we need to delete copy and move ctors.
  IBVerbsCommunicator(const IBVerbsCommunicator&) noexcept = delete;
  IBVerbsCommunicator& operator=(const IBVerbsCommunicator&) noexcept = delete;

  // move
  IBVerbsCommunicator(IBVerbsCommunicator&&) noexcept = delete;
  IBVerbsCommunicator& operator=(IBVerbsCommunicator&&) noexcept = delete;

  // connection management
  struct ProcessInfo RegisterProcess(int dest_rank, struct ProcessInfo pinfo);
  struct ProcessInfo CreateQueuePair(int dest_rank);
  void RegisterQueuePair(int dest_rank, struct ProcessInfo pinfo);
  void RegisterMyself(int my_rank);

  void Send(int dest_rank, const void* buf, size_t len, bool blocking = true);
  void Recv(int src_rank, void* buf, size_t len, bool blocking = true);

  // wait ( for non-blocking io )
  bool SendPoll(int dest_rank);
  bool RecvPoll(int src_rank);
  void SendWait(int dest_rank);
  void RecvWait(int src_rank);

  // allreduce
  template <typename T>
  void AllreduceRing(const T* sendbuf, T* recvbuf, size_t len_elements);

  template <typename T>
  void AllreduceRabenseifner(const T* sendbuf, T* recvbuf, size_t len_elements);

  template <typename T>
  void AllreduceRingCuda(const T* sendbuf, T* recvbuf, size_t len_elements);

  template <typename T>
  void AllreduceRabenseifnerCuda(const T* sendbuf, T* recvbuf, size_t len_elements);

  int my_rank_;
  size_t world_size_;
};
```

## Error Codes
Error codes are defined at `ibcomm/util.h`.
When error occurs, ibcomm returns these values as an exit code.

```cpp
enum class IBCOMM_ERROR_CODE : int {
    INVALID_ARGUMENT = 1,

    // Error occured in InfiniBand Verbs call.
    IBVERBS_ERROR = 2,

    // Error occured in CUDA call.
    CUDA_ERROR = 3,

    NOT_SUPPORTED = 4
};
```

# How to run unit tests
## Setup allreduce integration tests
Integration tests of allreduce routines are implemented using `pytest` module.

```
$ pip install pytest
```

Unit tests depend on a few external libraries.

```
$ cd `Your cloned directory`
$ git submodule init
$ git submodule update
$ mkdir -p build
$ cd build
$ cmake ..
$ make -j

# Make sure `allreduce_tester` is generated
```

## Run allreduce integration tests
```
$ cd `Your cloned directory`
$ pytest
$ export HOSTFILE=hostfile          # Optional
$ pytest --capture=no               # For more info
$ pytest --capture=no -m "not slow" # Skip aging test
```

## Setup and run C++ unit tests
C++ unit tests depend on Google test (https://github.com/google/googletest).

First, download and build Google test.

```
$ WORKING_DIR=/tmp  # Your favorite directory
$ cd ${WORKING_DIR}
$ git clone https://github.com/google/googletest.git
$ cd googletest
$ mkdir build
$ cmake ..

$ cd `Your cloned directory`
$ cd build
$ cmake -D GOOGLETEST_ROOT=${WORKING_DIR}/googletest ..
$ make
$ ./unittest

Running main() from 
[==========] Running 4 tests from 1 test case.
[----------] Global test environment set-up.
[----------] 4 tests from IBCommUtilTest
[ RUN      ] IBCommUtilTest.ParseNumberZero
[       OK ] IBCommUtilTest.ParseNumberZero (0 ms)
[ RUN      ] IBCommUtilTest.ParseNumberPositive
[       OK ] IBCommUtilTest.ParseNumberPositive (0 ms)
[ RUN      ] IBCommUtilTest.ParseNumberMalformed
[       OK ] IBCommUtilTest.ParseNumberMalformed (0 ms)
[ RUN      ] IBCommUtilTest.get_exp_of_two
[       OK ] IBCommUtilTest.get_exp_of_two (0 ms)
[----------] 4 tests from IBCommUtilTest (0 ms total)

[----------] Global test environment tear-down
[==========] 4 tests from 1 test case ran. (0 ms total)
[  PASSED  ] 4 tests.
```

# Coding guideline
We adopt Google C++ Style Guide ( https://google.github.io/styleguide/cppguide.html ).

```
$ pip install cpplint
$ cpplint --recursive .
```

# Acknowledgements
We would like to thank Mr. Minoru Nakamura for his comprehensive document 
on Infiniband Verbs API. (http://www.nminoru.jp/~nminoru/network/infiniband/) (In Japanese)

# LICENSE
MIT License
