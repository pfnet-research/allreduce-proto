// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#pragma once

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <queue>
#include <vector>

#include "ibcomm/ibverbs_communicator.h"
#include "ibcomm/memory_pool.h"
#include "ibcomm/util.h"

#ifdef USE_TRACE
#define TRACE(NAME) util::trace(&NAME);
#else
#define TRACE(NAME)
#endif

#define THREADS 512

template <typename T>
__global__ void _reduce_inplace_cuda(T* result, const T* value,
                                     size_t len_elements) {
#pragma unroll
  for (auto index = blockDim.x * blockIdx.x + threadIdx.x; index < len_elements;
       index += blockDim.x * gridDim.x)
    result[index] += value[index];
}

template <typename T>
void IBVerbsCommunicator::AllreduceRingCuda(const T* sendbuf, T* recvbuf,
                                            size_t len_elements) {
  TRACE(trace_other_);

  if (world_size_ == 1) {
    CUDACHECK(cudaMemcpy(recvbuf, sendbuf, sizeof(T) * len_elements,
                         cudaMemcpyDefault));
    TRACE(trace_other_);

    return;
  }

  auto ranges = SplitBuffer(len_elements, sizeof(T));
  auto rank_to_chunk = GetRankToChunk(ranges);

  auto chunks = ranges.size();

  int from_rank = (my_rank_ - 1 + world_size_) % world_size_;
  int to_rank = (my_rank_ + 1) % world_size_;

  auto controller =
      pool_->GetController((ranges[0].second - ranges[0].first) * sizeof(T));

  std::vector<Memory*> chunk_to_memory(chunks, NULL);

  std::queue<int> reg_q;
  // Reduce-Scatter's recv
  for (int i = 0; i < world_size_ - 1; i++) {
    // [my_rank_ - 1, my_rank_)
    int rank = (my_rank_ - i - 1 + world_size_) % world_size_;

    for (auto it = rank_to_chunk[rank].rbegin();
         it != rank_to_chunk[rank].rend(); ++it) {
      reg_q.push(*it);
    }
  }

  // AllGather's recv
  for (int i = 0; i < world_size_ - 1; i++) {
    // [my_rank_, my_rank_ -1, ...,  my_rank_ + 1)
    int rank = (my_rank_ - i + world_size_) % world_size_;
    for (auto it = rank_to_chunk[rank].rbegin();
         it != rank_to_chunk[rank].rend(); ++it) {
      reg_q.push(*it);
    }
  }

  std::queue<int> first_send_q;
  std::queue<int> first_send_q_buffering;

  for (auto it = rank_to_chunk[my_rank_].rbegin();
       it != rank_to_chunk[my_rank_].rend(); ++it) {
    first_send_q.push(*it);
  }

  int current_recv_i = reg_q.front();

  std::queue<int> wait_send_q;
  std::queue<int> wait_reduction_q;
  std::queue<int> wait_send_completion_q;
  int remaining_recv_q_length = 0;
  bool reduce_scatter_phase = true;
  // last rank (end of allgather)
  const int final_rank = (my_rank_ + 2) % world_size_;

  TRACE(trace_other_);

  while (true) {
    while ((reduce_scatter_phase || wait_reduction_q.empty()) &&
           RecvPoll(from_rank)) {
      TRACE(trace_received_);

      remaining_recv_q_length--;
      auto range = ranges[current_recv_i];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      using util::ceilDiv;

      const auto blocks =
          std::min(ceilDiv(elements, (size_t)THREADS), (size_t)(65535));

      auto& mem = chunk_to_memory[current_recv_i];

      TRACE(trace_received_);

      if (reduce_scatter_phase) {
        TRACE(trace_issue_redu_kernel_);
      } else {
        TRACE(trace_issue_copy_kernel_);
      }

      CUDACHECK(cudaMemcpyAsync(recvbuf + offset_elements, mem->ptr(), bytes,
                                cudaMemcpyDefault, mem->stream()));
      if (reduce_scatter_phase) {
        _reduce_inplace_cuda<<<blocks, THREADS, 0, mem->stream()>>>(
            recvbuf + offset_elements, sendbuf + offset_elements, elements);
        CUDACHECK(cudaMemcpyAsync(mem->ptr(), recvbuf + offset_elements, bytes,
                                  cudaMemcpyDefault, mem->stream()));
        wait_reduction_q.push(current_recv_i);
        if (current_recv_i == rank_to_chunk[to_rank].front()) {
          reduce_scatter_phase = false;
        }

        TRACE(trace_issue_redu_kernel_);
      } else {
        TRACE(trace_issue_copy_kernel_);
        if (current_recv_i < rank_to_chunk[final_rank].front() ||
            rank_to_chunk[final_rank].back() < current_recv_i) {
          TRACE(trace_issue_send_);

          SendRegistered(to_rank, mem->ptr(), mem->mr(), bytes, false);
          wait_send_completion_q.push(current_recv_i);

          TRACE(trace_issue_send_);
        } else {
          // NO NEED SEND because this is last allgather step.
          CUDACHECK(cudaStreamSynchronize(mem->stream()));
          controller.returnMemory(mem);
          mem = NULL;
        }
      }
      current_recv_i = (current_recv_i - 1 + chunks) % chunks;
    }

    // This means rank_to_chunk[final_rank].front() == current_recv_i in
    // RecvPoll loop (current_recv_i is already decremented)
    if (!reduce_scatter_phase && wait_reduction_q.empty() &&
        current_recv_i ==
            (rank_to_chunk[final_rank].front() - 1 + chunks) % chunks)
      break;  // DONE!

    if (!wait_reduction_q.empty() &&
        cudaStreamQuery(chunk_to_memory[wait_reduction_q.front()]->stream()) ==
            cudaSuccess) {
      TRACE(trace_reduced_);

      int i = wait_reduction_q.front();
      wait_reduction_q.pop();

      auto range = ranges[i];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      TRACE(trace_reduced_);
      // This send is reduce-scatter phase send and allgather phase first send,
      // thus we can send all chunk.
      if (first_send_q.empty() && wait_send_q.empty() &&
          first_send_q_buffering.empty()) {
        TRACE(trace_issue_send_);

        auto mem = chunk_to_memory[i];
        SendRegistered(to_rank, mem->ptr(), mem->mr(), bytes, false);
        wait_send_completion_q.push(i);

        TRACE(trace_issue_send_);
      } else {
        first_send_q_buffering.push(i);
      }
    }

    // When first_send is not completed, We cannot issue first-send's recv.
    if (remaining_recv_q_length <= 2 && !reg_q.empty() &&
        chunk_to_memory[reg_q.front()] == NULL) {
      TRACE(trace_issue_recv_);

      int recv_key = reg_q.front();
      reg_q.pop();

      auto range = ranges[recv_key];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[recv_key] = controller.getMemory();

      remaining_recv_q_length++;
      RecvRegistered(from_rank, mem->ptr(), mem->mr(), bytes, false);

      TRACE(trace_issue_recv_);
    }

    if (!first_send_q.empty()) {
      TRACE(trace_issue_copy_kernel_);

      int i = first_send_q.front();
      first_send_q.pop();

      auto range = ranges[i];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[i] = controller.getMemory();
      CUDACHECK(cudaMemcpyAsync(mem->ptr(), sendbuf + offset_elements, bytes,
                                cudaMemcpyDefault, mem->stream()));
      wait_send_q.push(i);

      TRACE(trace_issue_copy_kernel_);
    }

    if (!wait_send_q.empty() &&
        cudaStreamQuery(chunk_to_memory[wait_send_q.front()]->stream()) ==
            cudaSuccess) {
      TRACE(trace_issue_send_);

      int i = wait_send_q.front();
      wait_send_q.pop();

      auto range = ranges[i];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[i];

      SendRegistered(to_rank, mem->ptr(), mem->mr(), bytes, false);

      wait_send_completion_q.push(i);

      TRACE(trace_issue_send_);
    } else if (first_send_q.empty() && wait_send_q.empty() &&
               !first_send_q_buffering.empty()) {
      TRACE(trace_issue_send_);
      while (!first_send_q_buffering.empty()) {
        int i = first_send_q_buffering.front();
        first_send_q_buffering.pop();

        auto range = ranges[i];
        size_t elements = (range.second - range.first);
        size_t bytes = elements * sizeof(T);

        auto mem = chunk_to_memory[i];

        SendRegistered(to_rank, mem->ptr(), mem->mr(), bytes, false);
        wait_send_completion_q.push(i);
      }
      TRACE(trace_issue_send_);
    }

    while (SendPoll(to_rank)) {
      TRACE(trace_other_);

      int complete_send_chunk_id = wait_send_completion_q.front();
      wait_send_completion_q.pop();
      CUDACHECK(cudaStreamSynchronize(
          chunk_to_memory[complete_send_chunk_id]->stream()));
      controller.returnMemory(chunk_to_memory[complete_send_chunk_id]);
      chunk_to_memory[complete_send_chunk_id] = NULL;

      TRACE(trace_other_);
    }
  }

  TRACE(trace_other_);
  while (!wait_send_completion_q.empty()) {
    SendWait(to_rank);
    int complete_send_chunk_id = wait_send_completion_q.front();
    wait_send_completion_q.pop();
    // We need sync because memcpy is issued.
    CUDACHECK(cudaStreamSynchronize(
        chunk_to_memory[complete_send_chunk_id]->stream()));
    controller.returnMemory(chunk_to_memory[complete_send_chunk_id]);
    chunk_to_memory[complete_send_chunk_id] = NULL;
  }
  TRACE(trace_other_);
}

class Chunk {
 public:
  int range_id_;

  int depth_;

  // pair_rank_ is no meaning in some context.
  int pair_rank_;

  // reduce_ is no meaning in some context.
  bool reduce_;

  bool last_;

  Chunk(int range_id, int depth, int pair_rank = -1, bool reduce = false,
        bool last = false)
      : range_id_(range_id),
        depth_(depth),
        pair_rank_(pair_rank),
        reduce_(reduce),
        last_(last) {}
};

template <typename T>
void IBVerbsCommunicator::AllreduceRabenseifnerCuda(const T* sendbuf,
                                                    T* recvbuf,
                                                    size_t len_elements) {
  TRACE(trace_other_);
  bool first_memcpy_done = false;
  CUDACHECK(cudaMemcpyAsync(recvbuf, sendbuf, sizeof(T) * len_elements,
                            cudaMemcpyDefault));

  if (world_size_ == 1) {
    CUDACHECK(cudaStreamSynchronize(0));

    TRACE(trace_other_);
    return;
  }

  int world_size_exp = util::GetExpOfTwo(world_size_);

  // check world_size is power-of-2 or not
  if (world_size_exp == 0) {
    TRACE(trace_other_);
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
                      "Currently, rabenseifner's algorithm doesn't support "
                      "non-power-of-2 processes.");
  }

  auto ranges = SplitBuffer(len_elements, sizeof(T));
  auto rank_to_chunk = GetRankToChunk(ranges);

  auto chunks = ranges.size();

  auto controller =
      pool_->GetController((ranges[0].second - ranges[0].first) * sizeof(T));
  std::vector<Memory*> chunk_to_memory(chunks, NULL);

  std::vector<bool> reduced_chunk(chunks, false);
  std::queue<Chunk> recv_q;
  std::queue<Chunk> wait_recv_q;
  std::queue<Chunk> send_q;
  std::queue<Chunk> wait_send_q;
  std::queue<Chunk> wait_send_copy_q;
  std::queue<Chunk> wait_reduction_q;
  std::queue<Chunk> first_send_q;
  std::queue<Chunk> first_send_q_buffering;

  // GPU working memory size check and realloc if need
  if (tmp_gpu_buffer_size_ < sizeof(T) * (ranges[0].second - ranges[0].first)) {
    // dealloc
    util::IbcommWarning(
        __FILE__, __LINE__,
        "IBCOMM_GPU_WORK_MEMORY_SIZE is smaller than chunk size.\n"
        "runtime-reallocation is occured.");
    CUDACHECK(cudaFree(tmp_gpu_buffer_));

    // alloc
    tmp_gpu_buffer_size_ = sizeof(T) * (ranges[0].second - ranges[0].first);
    CUDACHECK(cudaMalloc(static_cast<void**>(&tmp_gpu_buffer_),
                         tmp_gpu_buffer_size_));
  }

  int start_rank = 0;
  int end_rank = world_size_;

  // Reduce-Scatter (recursive halving)
  for (int step = 0; step < world_size_exp; step++) {
    int to_rank = my_rank_ ^ (1 << step);

    int send_rank_start, send_rank_end, recv_rank_start, recv_rank_end;
    if (my_rank_ < to_rank) {
      // I send front rank
      send_rank_start = start_rank;
      send_rank_end = end_rank - (end_rank - start_rank) / 2;
      recv_rank_start = send_rank_end;
      recv_rank_end = end_rank;
    } else {
      // I send back rank
      recv_rank_start = start_rank;
      recv_rank_end = end_rank - (end_rank - start_rank) / 2;
      send_rank_start = recv_rank_end;
      send_rank_end = end_rank;
    }

    for (int recv_rank = recv_rank_start; recv_rank < recv_rank_end;
         recv_rank++) {
      for (auto chunk : rank_to_chunk[recv_rank]) {
        recv_q.emplace(chunk, step, to_rank, true);
      }
    }
    for (int send_rank = send_rank_start; send_rank < send_rank_end;
         send_rank++) {
      for (auto chunk : rank_to_chunk[send_rank]) {
        if (step == 0) {
          first_send_q.emplace(chunk, step, to_rank);
        } else {
          send_q.emplace(chunk, step, to_rank);
        }
      }
    }

    start_rank = recv_rank_start;
    end_rank = recv_rank_end;
  }

  // AllGather (recursive doubling)
  for (int step = 0; step < world_size_exp; step++) {
    int to_rank = my_rank_ ^ (1 << (world_size_exp - step - 1));

    int send_rank_start, send_rank_end, recv_rank_start, recv_rank_end;
    if (my_rank_ > to_rank) {
      // I send front rank
      send_rank_start = start_rank;
      send_rank_end = end_rank;
      recv_rank_start = send_rank_end;
      recv_rank_end = recv_rank_start + end_rank - start_rank;
    } else {
      // I send back rank
      send_rank_start = start_rank;
      send_rank_end = end_rank;
      recv_rank_end = send_rank_start;
      recv_rank_start = recv_rank_end - (end_rank - start_rank);
    }

    for (int recv_rank = recv_rank_start; recv_rank < recv_rank_end;
         recv_rank++) {
      for (auto chunk : rank_to_chunk[recv_rank]) {
        recv_q.emplace(chunk, step + world_size_exp, to_rank);
      }
    }

    for (int send_rank = send_rank_start; send_rank < send_rank_end;
         send_rank++) {
      for (auto chunk : rank_to_chunk[send_rank]) {
        send_q.emplace(chunk, step + world_size_exp, to_rank, true,
                       step == (world_size_exp - 1));
      }
    }

    start_rank = std::min(send_rank_start, recv_rank_start);
    end_rank = std::max(send_rank_end, recv_rank_end);
  }

  TRACE(trace_other_);

  while (!recv_q.empty() || !wait_recv_q.empty() || !send_q.empty() ||
         !wait_send_q.empty() || !wait_send_copy_q.empty() ||
         !wait_reduction_q.empty() || !first_send_q.empty() ||
         !first_send_q_buffering.empty()) {
    while (wait_reduction_q.empty() && !wait_recv_q.empty() &&
           RecvPoll(wait_recv_q.front().pair_rank_)) {
      TRACE(trace_received_);

      auto received = wait_recv_q.front();
      wait_recv_q.pop();

      auto range = ranges[received.range_id_];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto& mem = chunk_to_memory[received.range_id_];

      TRACE(trace_received_);

      if (received.reduce_) {
        // Reduce-Scatter phase

        if (!first_memcpy_done) {
          TRACE(trace_other_);
          CUDACHECK(cudaStreamSynchronize(0));
          first_memcpy_done = true;
          TRACE(trace_other_);
        }

        TRACE(trace_issue_redu_kernel_);

        // tmp_gpu_buffer <- mem
        CUDACHECK(cudaMemcpyAsync(tmp_gpu_buffer_, mem->ptr(), bytes,
                                  cudaMemcpyDefault, mem->stream()));

        const auto blocks =
            std::min(util::ceilDiv(elements, (size_t)THREADS), (size_t)(65535));

        // recvbuf += tmp_gpu_buffer ( on GPU )
        _reduce_inplace_cuda<<<blocks, THREADS, 0, mem->stream()>>>(
            recvbuf + offset_elements, static_cast<T*>(tmp_gpu_buffer_),
            elements);

        // mem <- recvbuf
        CUDACHECK(cudaMemcpyAsync(mem->ptr(), recvbuf + offset_elements, bytes,
                                  cudaMemcpyDefault, mem->stream()));

        received.depth_++;
        wait_reduction_q.push(received);

        TRACE(trace_issue_redu_kernel_);
      } else {
        // AllGather phase

        TRACE(trace_issue_copy_kernel_);

        // recvbuf <- mem
        CUDACHECK(cudaMemcpyAsync(recvbuf + offset_elements, mem->ptr(), bytes,
                                  cudaMemcpyDefault, mem->stream()));

        reduced_chunk[received.range_id_] = true;

        TRACE(trace_issue_copy_kernel_);
      }
    }

    if (!wait_reduction_q.empty() &&
        cudaStreamQuery(
            chunk_to_memory[wait_reduction_q.front().range_id_]->stream()) ==
            cudaSuccess) {
      TRACE(trace_reduced_);

      auto reduced = wait_reduction_q.front();
      wait_reduction_q.pop();

      auto range = ranges[reduced.range_id_];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto& mem = chunk_to_memory[reduced.range_id_];

      TRACE(trace_reduced_);

      if (!send_q.empty() && send_q.front().range_id_ == reduced.range_id_ &&
          send_q.front().depth_ == reduced.depth_) {
        auto send_range = send_q.front();
        send_q.pop();

        if (first_send_q.empty() && wait_send_copy_q.empty() &&
            first_send_q_buffering.empty()) {
          TRACE(trace_issue_send_);

          SendRegistered(send_range.pair_rank_, mem->ptr(), mem->mr(), bytes,
                         false);
          wait_send_q.push(send_range);

          TRACE(trace_issue_send_);
        } else {
          first_send_q_buffering.push(send_range);
        }

        if (send_range.reduce_) {
          // AllGather phase
          reduced_chunk[send_range.range_id_] = true;
        }
      } else {
        CUDACHECK(cudaStreamSynchronize(mem->stream()));
        controller.returnMemory(mem);
        mem = NULL;
      }
    }

    if (wait_recv_q.size() <= 2 && !recv_q.empty() &&
        chunk_to_memory[recv_q.front().range_id_] == NULL) {
      TRACE(trace_issue_recv_);

      auto recv_range = recv_q.front();
      recv_q.pop();

      auto range = ranges[recv_range.range_id_];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[recv_range.range_id_] = controller.getMemory();

      RecvRegistered(recv_range.pair_rank_, mem->ptr(), mem->mr(), bytes,
                     false);
      wait_recv_q.push(recv_range);

      TRACE(trace_issue_recv_);
    }

    while (first_send_q.empty() && wait_send_copy_q.empty() &&
           first_send_q_buffering.empty() && !send_q.empty() &&
           reduced_chunk[send_q.front().range_id_]) {
      TRACE(trace_issue_send_);

      auto send_range = send_q.front();
      send_q.pop();

      auto range = ranges[send_range.range_id_];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[send_range.range_id_];

      SendRegistered(send_range.pair_rank_, mem->ptr(), mem->mr(), bytes,
                     false);
      wait_send_q.push(send_range);

      TRACE(trace_issue_send_);
    }

    if (!first_send_q.empty()) {
      TRACE(trace_issue_copy_kernel_);

      auto send_range = first_send_q.front();
      first_send_q.pop();

      auto range = ranges[send_range.range_id_];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[send_range.range_id_] = controller.getMemory();
      CUDACHECK(cudaMemcpyAsync(mem->ptr(), sendbuf + offset_elements, bytes,
                                cudaMemcpyDefault, mem->stream()));

      wait_send_copy_q.push(send_range);

      TRACE(trace_issue_copy_kernel_);
    }

    if (!wait_send_copy_q.empty() &&
        cudaStreamQuery(
            chunk_to_memory[wait_send_copy_q.front().range_id_]->stream()) ==
            cudaSuccess) {
      TRACE(trace_issue_send_);

      auto send_range = wait_send_copy_q.front();
      wait_send_copy_q.pop();

      auto range = ranges[send_range.range_id_];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[send_range.range_id_];

      SendRegistered(send_range.pair_rank_, mem->ptr(), mem->mr(), bytes,
                     false);
      wait_send_q.push(send_range);

      TRACE(trace_issue_send_);
    }

    while (first_send_q.empty() && wait_send_copy_q.empty() &&
           !first_send_q_buffering.empty()) {
      TRACE(trace_issue_send_);

      auto send_range = first_send_q_buffering.front();
      first_send_q_buffering.pop();

      auto range = ranges[send_range.range_id_];
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mem = chunk_to_memory[send_range.range_id_];

      SendRegistered(send_range.pair_rank_, mem->ptr(), mem->mr(), bytes,
                     false);
      wait_send_q.push(send_range);

      TRACE(trace_issue_send_);
    }

    while (!wait_send_q.empty() && SendPoll(wait_send_q.front().pair_rank_)) {
      TRACE(trace_other_);

      auto send_range = wait_send_q.front();
      wait_send_q.pop();

      if (send_range.reduce_ && !send_range.last_) {
        // AllGather phase and non-last AllGather send
        // We need to send a chunk which is already sent,
        // so we still hold data on CPU-memory.
      } else {
        auto& mem = chunk_to_memory[send_range.range_id_];

        CUDACHECK(cudaStreamSynchronize(mem->stream()));
        controller.returnMemory(mem);
        mem = NULL;
      }

      TRACE(trace_other_);
    }
  }

  TRACE(trace_other_);
  for (auto& mem : chunk_to_memory) {
    if (mem != NULL) {
      CUDACHECK(cudaStreamSynchronize(mem->stream()));
      controller.returnMemory(mem);
      mem = NULL;
    }
  }
  TRACE(trace_other_);
}

#endif
