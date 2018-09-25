// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#pragma once

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <queue>
#include <vector>

#include "ibcomm/util.h"

class IBVerbsCommunicator;

/* Concept:
 * MemoryPool : manages (raw-) `MemoryBlock`. unit of `ibv_reg_mr`.
 * MemoryController : manages `Memory`. unit of `cudaStream`.
 * MemoryAllocator : allocates `MemoryBlock`.
 *  MemoryAllocator(size_t initial_size, IBVerbsCommunicator& comm);
 *  std::unique_ptr<MemoryBlock> Allocate();
 * MemoryBlock : memory.
 * Memory : chunk.
 */

class MemoryBlock {
 public:
  // ctor
  MemoryBlock(size_t size, IBVerbsCommunicator* comm);

  // Manages raw pointers thus we need to delete copy and move ctors.
  // copy
  MemoryBlock(const MemoryBlock&) noexcept = delete;
  MemoryBlock& operator=(const MemoryBlock&) noexcept = delete;

  // move
  MemoryBlock(MemoryBlock&&) noexcept = delete;
  MemoryBlock& operator=(MemoryBlock&&) noexcept = delete;

  ~MemoryBlock();

  inline void* ptr() { return ptr_; }
  inline size_t length() const { return length_; }
  inline struct ibv_mr* mr() { return mr_; }

 private:
  IBVerbsCommunicator& comm_;

  void* ptr_;
  size_t length_;
  struct ibv_mr* mr_;
};

class Memory {
 public:
  Memory(MemoryBlock* block, size_t offset);

  inline void* ptr() {
    return static_cast<void*>(static_cast<char*>(block_.ptr()) + offset_);
  }
  inline cudaStream_t stream() { return stream_; }
  inline struct ibv_mr* mr() { return block_.mr(); }
  Memory* SetStream(cudaStream_t stream);
  Memory* UnsetStream();

 private:
  MemoryBlock& block_;
  size_t offset_;
  cudaStream_t stream_;
};

template <class MemoryAllocator>
class MemoryController;

template <class MemoryAllocator>
class MemoryPool {
  friend class MemoryController<MemoryAllocator>;

 public:
  static constexpr int DefaultMaxNumCudaStream = 128;
  static constexpr int DefaultPreAllocSize = 64 * 1024 * 1024;  // 64 MB.

  // ctor
  explicit MemoryPool(IBVerbsCommunicator* comm)
      : comm_(*comm),
        cuda_streams_(ReadNumCudaStream()),
        allocator_(ReadPreAllocSize(), comm) {
    if (comm == nullptr)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "comm is nullptr.");

    for (auto& stream : cuda_streams_) {
      CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }

    Allocate();
  }

  // Manages cudaStream_t thus we need to delete copy and move ctors.
  // copy
  MemoryPool(const MemoryPool&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;

  // move
  MemoryPool(MemoryPool&&) = delete;
  MemoryPool& operator=(MemoryPool&&) = delete;

  MemoryController<MemoryAllocator> GetController(size_t chunk_size) {
    if (controller_in_use_) {
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
                        "Currently, MemoryController is in use.");
    }

    controller_in_use_ = true;
    return MemoryController<MemoryAllocator>(this, chunk_size, cuda_streams_,
                                             memory_blocks_);
  }

  ~MemoryPool() {
    for (auto& stream : cuda_streams_) {
      cudaStreamDestroy(stream);
    }
  }

 private:
  IBVerbsCommunicator& comm_;

  MemoryAllocator allocator_;
  std::vector<std::unique_ptr<MemoryBlock>> memory_blocks_;
  std::vector<cudaStream_t> cuda_streams_;
  bool controller_in_use_ = false;

  // Read NumCudaStream from environmental variable.
  // Returns the default size `DefaultMaxNumCudaStream` if it is not set.
  int ReadNumCudaStream() {
    const char* envvar = getenv("IBCOMM_NUM_CUDA_STREAM");
    if (envvar) {
      int n = atoi(envvar);

      if (n <= 0 || n > 1024) {
        util::IbcommError(
            __FILE__, __LINE__, util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
            "Invalid value for IBCOMM_NUM_CUDA_STREAM: %s", envvar);
      }

      return n;
    } else {
      return DefaultMaxNumCudaStream;
    }
  }

  // Read PreAllocateSize from environmental variable.
  // Returns the default size `DefaultPreAllocSize` if it is not set.
  int ReadPreAllocSize() {
    const char* envvar = getenv("IBCOMM_MEMORY_POOL_PRE_ALLOC");
    if (envvar) {
      int n = atoi(envvar);

      if (n < 4 || n > 1 * 1024 * 1024 * 1024) {
        util::IbcommError(
            __FILE__, __LINE__, util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
            "Invalid value for IBCOMM_MEMORY_POOL_PRE_ALLOC: %s", envvar);
      }

      return n;
    } else {
      return DefaultPreAllocSize;
    }
  }

  void CompleteMemoryController() { controller_in_use_ = false; }

  std::unique_ptr<MemoryBlock>& Allocate() {
    memory_blocks_.push_back(allocator_.Allocate());
    return memory_blocks_.back();
  }

  cudaStream_t AddCudaStream() {
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_streams_.push_back(stream);

    return stream;
  }
};

template <class MemoryAllocator>
class MemoryController {
 public:
  MemoryController(MemoryPool<MemoryAllocator>* pool, size_t chunk_size,
                   const std::vector<cudaStream_t>& streams,
                   const std::vector<std::unique_ptr<MemoryBlock>>& blocks)
      : pool_(*pool),
        chunk_size_(chunk_size),
        streams_(streams),
        blocks_(blocks) {
    if (pool == nullptr)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "pool is nullptr.");

    for (auto stream : streams_) vacant_streams_.push(stream);

    for (const auto& block : blocks_) {
      AddMemoryBlockToVacantMemories(block.get());
    }
  }

  // Manages Memory thus we need to delete copy ctors.
  // copy
  MemoryController(const MemoryController&) = delete;
  MemoryController& operator=(const MemoryController&) = delete;

  // move
  MemoryController(MemoryController&&) = default;
  MemoryController& operator=(MemoryController&&) = default;

  Memory* getMemory() {
    if (vacant_streams_.empty()) {
      vacant_streams_.push(pool_.AddCudaStream());
    }
    cudaStream_t stream = vacant_streams_.front();
    vacant_streams_.pop();

    if (vacant_memories_.empty()) {
      AddMemoryBlockToVacantMemories(pool_.Allocate().get());
    }
    Memory* memory = vacant_memories_.front();
    vacant_memories_.pop();

    return memory->SetStream(stream);
  }

  void returnMemory(Memory* memory) {
    auto stream = memory->stream();
    vacant_memories_.push(memory->UnsetStream());
    vacant_streams_.push(stream);
  }

  ~MemoryController() {
    while (!vacant_memories_.empty()) {
      auto memory = vacant_memories_.front();
      vacant_memories_.pop();

      delete memory;
    }
    pool_.CompleteMemoryController();
  }

 private:
  MemoryPool<MemoryAllocator>& pool_;
  size_t chunk_size_;
  const std::vector<std::unique_ptr<MemoryBlock>>& blocks_;
  const std::vector<cudaStream_t>& streams_;

  std::queue<cudaStream_t> vacant_streams_;
  std::queue<Memory*> vacant_memories_;

  void AddMemoryBlockToVacantMemories(MemoryBlock* block) {
    if (block == nullptr)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "block is nullptr.");

    for (int i = 0; (i + 1) * chunk_size_ < block->length(); i++) {
      vacant_memories_.push(new Memory(block, i * chunk_size_));
    }
  }
};

class ConstantMemoryAllocator {
 public:
  ConstantMemoryAllocator(size_t initial_size, IBVerbsCommunicator* comm);
  std::unique_ptr<MemoryBlock> Allocate();

 private:
  IBVerbsCommunicator& comm_;
  size_t size_;
};

#endif
