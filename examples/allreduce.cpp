// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#include <mpi.h>

#include <climits>
#include <iostream>
#include <random>
#include <vector>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#endif

#include "ibcomm/ibverbs_communicator.h"
#include "ibcomm/util.h"  // CUDACHECK

// 256 [MiB] of allreduce float32 vector
#define ARRAY_LENGTH 67108864

// processes per node
// PPN must be 1 in this prototype implementation
// When PPN is not 1, this implementation isn't optimized.
#define PPN 1

// warmup times
#define WARMUP 3

double inline GetTime() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);

  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

#ifdef USE_CUDA
  int intra_rank = mpi_rank % PPN;
  CUDACHECK(cudaSetDevice(intra_rank));
#endif

  IBVerbsCommunicator comm(mpi_size);

  std::vector<uint32_t> qps(mpi_size * 3);

  for (int i = 0; i < mpi_size; i++) {
    if (i == mpi_rank) {
      continue;
    }
    ProcessInfo pinfo = comm.CreateQueuePair(i);
    qps[i * 3 + 0] = pinfo.lid;
    qps[i * 3 + 1] = pinfo.qp_n;
    qps[i * 3 + 2] = pinfo.psn;
  }

  MPI_Alltoall(MPI_IN_PLACE, 3, MPI_UINT32_T, qps.data(), 3, MPI_UINT32_T,
               MPI_COMM_WORLD);

  for (int i = 0; i < mpi_size; i++) {
    if (i == mpi_rank) {
      comm.RegisterMyself(i);
    } else {
      ProcessInfo pinfo;
      pinfo.lid = qps[i * 3 + 0];
      pinfo.qp_n = qps[i * 3 + 1];
      pinfo.psn = qps[i * 3 + 2];
      comm.RegisterQueuePair(i, pinfo);
    }
  }

#ifdef USE_CUDA
  thrust::host_vector<float> sendbuf(ARRAY_LENGTH);
  thrust::host_vector<float> recvbuf(ARRAY_LENGTH);
#else
  std::vector<float> sendbuf(ARRAY_LENGTH);
  std::vector<float> recvbuf(ARRAY_LENGTH);
#endif

  // fixed seed
  std::mt19937 mt(0);
  // To avoid overflow, use short range of number.
  std::uniform_int_distribution<int> rand(-1000, 1000);

  for (int i = 0; i < ARRAY_LENGTH; i++) {
    sendbuf[i] = rand(mt);
  }

  std::vector<float> answer(ARRAY_LENGTH, 0);
  MPI_Allreduce(sendbuf.data(), answer.data(), ARRAY_LENGTH, MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);

  double start, end;

#ifdef USE_CUDA
  thrust::device_vector<float> gpu_sendbuf(ARRAY_LENGTH);
  thrust::device_vector<float> gpu_recvbuf(ARRAY_LENGTH);
  gpu_sendbuf = sendbuf;

  for (int i = 0; i < WARMUP + 1; i++) {
    start = GetTime();
    comm.AllreduceRingCuda(thrust::raw_pointer_cast(gpu_sendbuf.data()),
                           thrust::raw_pointer_cast(gpu_recvbuf.data()),
                           ARRAY_LENGTH);
    /*
     comm.AllreduceRabenseifnerCuda(thrust::raw_pointer_cast(gpu_sendbuf.data()),
                                   thrust::raw_pointer_cast(gpu_recvbuf.data()),
                                   ARRAY_LENGTH);
                                   */
    end = GetTime();
  }

  recvbuf = gpu_recvbuf;
#else
  for (int i = 0; i < WARMUP + 1; i++) {
    start = GetTime();
    comm.AllreduceRing(sendbuf.data(), recvbuf.data(), ARRAY_LENGTH);
    // comm.AllreduceRabenseifner(sendbuf.data(), recvbuf.data(), ARRAY_LENGTH);
    end = GetTime();
  }
#endif

  bool ok = true;
  for (int i = 0; i < ARRAY_LENGTH; i++) {
    if (fabs(answer[i] - recvbuf[i]) > 1e-6) {
      ok = false;
      std::cout << "wrong at " << mpi_rank << " rank " << i << " element "
                << answer[i] << ":" << recvbuf[i] << std::endl;
    } else {
      //            std::cerr << "ok at " << mpi_rank << " rank " << i << "
      //            element " << answer[i] << ":" << recvbuf[i] << std::endl;
    }
  }

  std::cout << "rank: " << mpi_rank << (ok ? " OK" : " FAIL") << std::endl;

  MPI_Finalize();

  if (mpi_rank == 0) printf("elapsed time : %e [s]\n", end - start);

  return 0;
}
