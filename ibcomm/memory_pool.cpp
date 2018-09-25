// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#include "ibcomm/memory_pool.h"

#include "ibcomm/ibverbs_communicator.h"
#include "ibcomm/util.h"

#ifdef USE_CUDA

// ~~~ Memory class ~~~ //
Memory::Memory(MemoryBlock* block, size_t offset)
    : block_(*block), offset_(offset) {
  if (block == nullptr)
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                      "block is nullptr.");
}

Memory* Memory::SetStream(cudaStream_t stream) {
  stream_ = stream;
  return this;
}

Memory* Memory::UnsetStream() { return SetStream(NULL); }

// ~~~ MemoryBlock class ~~~ //
MemoryBlock::MemoryBlock(size_t size, IBVerbsCommunicator* comm)
    : comm_(*comm), length_(size) {
  if (comm == nullptr)
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                      "comm is nullptr.");

  // not thread safe
  CUDACHECK(cudaHostAlloc(&ptr_, length_, cudaHostAllocDefault));
  mr_ = comm->RegisterRecvBuf(ptr_, length_);
}

MemoryBlock::~MemoryBlock() {
  ibv_dereg_mr(mr_);
  CUDACHECK(cudaFreeHost(ptr_));
}

ConstantMemoryAllocator::ConstantMemoryAllocator(size_t initial_size,
                                                 IBVerbsCommunicator* comm)
    : size_(initial_size), comm_(*comm) {
  if (comm == nullptr)
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                      "comm is nullptr.");
}

std::unique_ptr<MemoryBlock> ConstantMemoryAllocator::Allocate() {
  return std::unique_ptr<MemoryBlock>(new MemoryBlock(size_, &comm_));
}

#endif
