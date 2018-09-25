// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#pragma once

#include <infiniband/verbs.h>

#include <cstdint>
#include <ctime>

#include <queue>
#include <utility>
#include <vector>

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "ibcomm/memory_pool.h"

template <class MemoryAllocator>
class MemoryPool;
#endif

struct ProcessInfo {
  uint16_t lid;
  uint32_t qp_n;
  uint32_t psn;
};

struct ProcessQueue {
  struct ibv_cq* send_complete_queue;
  struct ibv_cq* recv_complete_queue;
  struct ibv_qp* queue_pair;

  ProcessQueue() {
    send_complete_queue = NULL;
    recv_complete_queue = NULL;
    queue_pair = NULL;
  }

  ProcessQueue(struct ibv_cq* scq, struct ibv_cq* rcq, struct ibv_qp* qp) {
    send_complete_queue = scq;
    recv_complete_queue = rcq;
    queue_pair = qp;
  }

  // copy
  ProcessQueue(const ProcessQueue&) = delete;
  ProcessQueue& operator=(const ProcessQueue&) = delete;

  // move
  // queues' are managed by IBVerbsCommunicator.
  ProcessQueue(ProcessQueue&&) noexcept = default;
  ProcessQueue& operator=(ProcessQueue&&) noexcept = default;
};

class Memory;

class IBVerbsCommunicator {
  // to export registerSendBuf, registerRecvBuf
  friend class MemoryBlock;

 public:
  // ctor
  IBVerbsCommunicator();
  explicit IBVerbsCommunicator(int world_size);

  // Manages infiniband-related resources thus we need to delete copy and move
  // ctors. copy
  IBVerbsCommunicator(const IBVerbsCommunicator&) noexcept = delete;
  IBVerbsCommunicator& operator=(const IBVerbsCommunicator&) noexcept = delete;

  // move
  IBVerbsCommunicator(IBVerbsCommunicator&&) noexcept = delete;
  IBVerbsCommunicator& operator=(IBVerbsCommunicator&&) noexcept = delete;

  // dtor
  ~IBVerbsCommunicator();

  // init
  void Init(int world_size);

  // connection management
  struct ProcessInfo RegisterProcess(int dest_rank, struct ProcessInfo pinfo);
  struct ProcessInfo CreateQueuePair(int dest_rank);
  void RegisterQueuePair(int dest_rank, struct ProcessInfo pinfo);
  void RegisterMyself(int my_rank);

  // send
  void Send(int dest_rank, const void* buf, size_t len, bool blocking = true);

  // recv
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

#ifdef USE_CUDA
  template <typename T>
  void AllreduceRingCuda(const T* sendbuf, T* recvbuf, size_t len_elements);

  template <typename T>
  void AllreduceRabenseifnerCuda(const T* sendbuf, T* recvbuf,
                                 size_t len_elements);

  void PrepareMemoryPool();
#endif

  // bcast
  void Bcast(void* buf, size_t len, int root);

  void SetTimerBase();
  void DumpTrace() const;

 private:
  bool initialized_ = false;
  struct ibv_port_attr port_attr_ = {};
  std::vector<ProcessQueue> pq_world_;
  std::vector<uint32_t> psn_world_;
  std::vector<std::pair<struct ibv_mr*, struct ibv_mr*>> mr_world_;

  struct ibv_mr* RegisterSendBuf(const void* buf, size_t len);
  void SendRegistered(int dest_rank, const void* buf, struct ibv_mr* mr_buf,
                      size_t len, bool blocking = true);

  struct ibv_mr* RegisterRecvBuf(void* buf, size_t len);
  void RecvRegistered(int src_rank, const void* buf, struct ibv_mr* mr_buf,
                      size_t len, bool blocking = true);

  void PopMrAndDereg(std::queue<struct ibv_mr*>* q);

#ifdef USE_CUDA
  std::unique_ptr<MemoryPool<ConstantMemoryAllocator>> pool_;

  void* tmp_gpu_buffer_ = NULL;
  size_t tmp_gpu_buffer_size_ = 0;
#endif

  // need destruction variables
  struct ibv_device** dev_list_ = NULL;
  struct ibv_context* context_ = NULL;

  // Protection Domain
  struct ibv_pd* pd_ = NULL;

  // local communication
  int my_rank_ = -1;
  size_t world_size_;

  // allreduce range func
  // Splits buffer based given chunk size
  std::vector<std::pair<size_t, size_t>> SplitBuffer(size_t len_elements,
                                                     size_t len_per_element);
  // Defines map (rank |-> chunk_ids)
  std::vector<std::vector<int>> GetRankToChunk(
      const std::vector<std::pair<size_t, size_t>>& ranges);

  struct timespec trace_start_;

  // receive is completed
  std::vector<struct timespec> trace_received_;

  // reduction is completed
  std::vector<struct timespec> trace_reduced_;

  // issue send
  std::vector<struct timespec> trace_issue_send_;

  // issue copy-kernel call
  std::vector<struct timespec> trace_issue_copy_kernel_;

  // issue reduce-kernel call
  std::vector<struct timespec> trace_issue_redu_kernel_;

  // issue recv
  std::vector<struct timespec> trace_issue_recv_;

  // others
  std::vector<struct timespec> trace_other_;
};

#include "ibcomm/allreduce_cpu_impl.h"
#include "ibcomm/allreduce_cuda_impl.h"
