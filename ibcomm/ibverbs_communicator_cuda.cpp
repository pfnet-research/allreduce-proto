// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#ifdef USE_CUDA
#include "ibcomm/ibverbs_communicator.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "ibcomm/memory_pool.h"

namespace {
int ReadWorkGpuMemorySize() {
  const char* size = getenv("IBCOMM_WORK_GPU_MEMORY_SIZE");

  if (size != NULL) {
    int size_int = atoi(size);

    return size_int;
  }

  return -1;  // use default size
}

};  // namespace

void IBVerbsCommunicator::PrepareMemoryPool() {
  pool_.reset(new MemoryPool<ConstantMemoryAllocator>(this));

  tmp_gpu_buffer_size_ = ReadWorkGpuMemorySize();

  if (tmp_gpu_buffer_size_ == -1) {
    tmp_gpu_buffer_size_ = 32 * 1024 * 1024;
  }

  CUDACHECK(
      cudaMalloc(static_cast<void**>(&tmp_gpu_buffer_), tmp_gpu_buffer_size_));
}

#endif
