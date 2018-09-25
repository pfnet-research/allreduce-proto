// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#pragma once

#include <cassert>

#include <algorithm>
#include <iostream>
#include <queue>
#include <utility>
#include <vector>

#include "ibcomm/ibverbs_communicator.h"
#include "ibcomm/util.h"

template <typename T>
void _reduce_inplace(T* result, const T* value, size_t len_elements) {
  for (size_t i = 0; i < len_elements; i++) {
    result[i] += value[i];
  }
}

template <typename T>
void IBVerbsCommunicator::AllreduceRing(const T* sendbuf, T* recvbuf,
                                        size_t len_elements) {
  if (world_size_ == 1) {
    memcpy(recvbuf, sendbuf, sizeof(T) * len_elements);
    return;
  }

  auto ranges = SplitBuffer(len_elements, sizeof(T));
  auto rank_to_chunk = GetRankToChunk(ranges);

  auto chunks = ranges.size();

  int from_rank = (my_rank_ - 1 + world_size_) % world_size_;
  int to_rank = (my_rank_ + 1) % world_size_;

  std::queue<int> reg_q;
  for (int i = 0; i < world_size_ - 1; i++) {
    // [my_rank_ - 1, my_rank_)
    int rank = (my_rank_ - i - 1 + world_size_) % world_size_;

    for (auto it = rank_to_chunk[rank].rbegin();
         it != rank_to_chunk[rank].rend(); ++it) {
      reg_q.push(*it);
    }
  }

  // send_mrs' mr needs SendPoll before deregistration
  // (used in ReduceScatter, AllGather).
  // However, recv_mrs' mr can deregistration immediately (used in AllGather).
  std::queue<struct ibv_mr*> send_mrs;
  std::queue<struct ibv_mr*> recv_mrs;

  // cached mrs
  std::vector<struct ibv_mr*> mrs(chunks, NULL);

  // HCA's Q length
  int send_q_elements = 0;
  int recv_q_elements = 0;

  std::queue<int> first_send_q;
  std::queue<int> first_send_q_buffering;

  for (auto it = rank_to_chunk[my_rank_].rbegin();
       it != rank_to_chunk[my_rank_].rend(); ++it) {
    first_send_q.push(*it);
  }

  int current_recv_i = reg_q.front();
  bool reduce_scatter_done = false;

  // ReduceScatter
  while (!reduce_scatter_done) {
    while ((reg_q.empty() || recv_q_elements > 0) && RecvPoll(from_rank)) {
      recv_q_elements--;

      auto range = ranges[current_recv_i];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      _reduce_inplace(recvbuf + offset_elements, sendbuf + offset_elements,
                      elements);

      if (current_recv_i == rank_to_chunk[to_rank].front()) {
        reduce_scatter_done = true;
      }

      if (!(rank_to_chunk[to_rank].front() <= current_recv_i &&
            current_recv_i <= rank_to_chunk[to_rank].back())) {
        if (first_send_q.empty() && first_send_q_buffering.empty()) {
          SendRegistered(to_rank, recvbuf + offset_elements,
                         mrs[current_recv_i], bytes, false);
          send_q_elements++;
        } else {
          first_send_q_buffering.push(current_recv_i);
        }
      }

      current_recv_i = (current_recv_i - 1 + chunks) % chunks;
    }

    if (!reg_q.empty()) {
      int recv_key = reg_q.front();
      reg_q.pop();

      auto range = ranges[recv_key];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      mrs[recv_key] = RegisterRecvBuf(recvbuf + offset_elements, bytes);

      RecvRegistered(from_rank, recvbuf + offset_elements, mrs[recv_key], bytes,
                     false);
      recv_q_elements++;
    }

    if (!first_send_q.empty()) {
      int i = first_send_q.front();
      first_send_q.pop();

      auto range = ranges[i];

      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      auto mr = RegisterSendBuf(sendbuf + offset_elements, bytes);

      SendRegistered(to_rank, sendbuf + offset_elements, mr, bytes, false);
      send_q_elements++;
      send_mrs.push(mr);
    } else {
      while (!first_send_q_buffering.empty()) {
        int i = first_send_q_buffering.front();
        first_send_q_buffering.pop();

        auto range = ranges[i];
        size_t offset_elements = range.first;
        size_t elements = (range.second - range.first);
        size_t bytes = elements * sizeof(T);

        SendRegistered(to_rank, recvbuf + offset_elements, mrs[i], bytes,
                       false);
        send_q_elements++;
      }
    }

    while (SendPoll(to_rank)) {
      send_q_elements--;
      if (!send_mrs.empty()) {
        PopMrAndDereg(&send_mrs);
      }
    }

    for (auto it = rank_to_chunk[my_rank_].begin();
         it != rank_to_chunk[my_rank_].end(); ++it) {
      auto range = ranges[*it];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      if (mrs[*it] == NULL) {
        mrs[*it] = RegisterRecvBuf(recvbuf + offset_elements, bytes);

        // when 1 chunk is registered, exit this loop to recv early.
        break;
      }
    }
  }

  // need sync before AllGather
  assert(recv_q_elements == 0);
  while (send_q_elements != 0) {
    SendWait(to_rank);
    send_q_elements--;

    if (!send_mrs.empty()) {
      PopMrAndDereg(&send_mrs);
    }
  }

  // AllGather
  for (int i = 0; i < world_size_; i++) {
    int rank = (1 + my_rank_ - i + world_size_) % world_size_;

    for (auto it = rank_to_chunk[rank].rbegin();
         it != rank_to_chunk[rank].rend(); ++it) {
      auto range = ranges[*it];
      size_t offset_elements = range.first;
      size_t elements = (range.second - range.first);
      size_t bytes = elements * sizeof(T);

      if (rank != (my_rank_ + 1) % world_size_) {
        RecvRegistered(from_rank, recvbuf + offset_elements, mrs[*it], bytes,
                       false);

        while (!RecvPoll(from_rank)) {
          if (SendPoll(to_rank)) {
            send_q_elements--;

            assert(!send_mrs.empty());

            PopMrAndDereg(&send_mrs);
          } else if (!recv_mrs.empty()) {
            PopMrAndDereg(&recv_mrs);
          }
        }
      }

      if (rank != (my_rank_ + 2) % world_size_) {
        SendRegistered(to_rank, recvbuf + offset_elements, mrs[*it], bytes,
                       false);
        send_mrs.push(mrs[*it]);
        send_q_elements++;
      } else {
        recv_mrs.push(mrs[*it]);
      }

      mrs[*it] = NULL;
    }
  }

  while (send_q_elements != 0) {
    SendWait(to_rank);
    send_q_elements--;

    if (!send_mrs.empty()) {
      PopMrAndDereg(&send_mrs);
    }
  }
  assert(send_mrs.empty());

  assert(recv_q_elements == 0);

  while (!recv_mrs.empty()) {
    PopMrAndDereg(&recv_mrs);
  }

  return;
}

template <typename T>
void IBVerbsCommunicator::AllreduceRabenseifner(const T* sendbuf, T* recvbuf,
                                                size_t len_elements) {
  if (world_size_ == 1) {
    memcpy(recvbuf, sendbuf, sizeof(T) * len_elements);
    return;
  }

  int world_size_exp = util::GetExpOfTwo(world_size_);

  // check world_size is power-of-2 or not
  if (world_size_exp == 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
                      "Currently, rabenseifner's algorithm doesn't support "
                      "non-power-of-2 processes.");
  }

  std::vector<std::pair<size_t, size_t>> ranges;
  for (int i = 0; i < world_size_; i++) {
    int range_length = util::ceilDiv(len_elements, world_size_);

    ranges.emplace_back(
        range_length * i,
        std::min(range_length * (i + 1), static_cast<int>(len_elements)));
  }

  T* tmp_buffer = static_cast<T*>(malloc(sizeof(T) * len_elements));
  if (tmp_buffer == NULL) {
    std::cerr << "Allocation of tmp-buffer failed" << std::endl;
    return;
  }

  memcpy(recvbuf, sendbuf, sizeof(T) * len_elements);

  // process maintains current chunk_range [start_chunk, end_chunk).
  int start_chunk = 0;
  int end_chunk = ranges.size();
  // Reduce-Scatter (recursive halving)
  for (int step = 0; step < world_size_exp; step++) {
    int to_rank = my_rank_ ^ (1 << step);

    int send_chunk_start, send_chunk_end, recv_chunk_start, recv_chunk_end;
    if (my_rank_ < to_rank) {
      // I send front chunk
      send_chunk_start = start_chunk;
      send_chunk_end = end_chunk - (end_chunk - start_chunk) / 2;
      recv_chunk_start = send_chunk_end;
      recv_chunk_end = end_chunk;
    } else {
      // I send back chunk
      recv_chunk_start = start_chunk;
      recv_chunk_end = end_chunk - (end_chunk - start_chunk) / 2;
      send_chunk_start = recv_chunk_end;
      send_chunk_end = end_chunk;
    }

    Send(to_rank, recvbuf + ranges[send_chunk_start].first,
         sizeof(T) *
             (ranges[(send_chunk_end - 1 + ranges.size()) % ranges.size()]
                  .second -
              ranges[send_chunk_start].first),
         false);

    Recv(to_rank, tmp_buffer + ranges[recv_chunk_start].first,
         sizeof(T) *
             (ranges[(recv_chunk_end - 1 + ranges.size()) % ranges.size()]
                  .second -
              ranges[recv_chunk_start].first),
         false);

    RecvWait(to_rank);

    _reduce_inplace(
        recvbuf + ranges[recv_chunk_start].first,
        tmp_buffer + ranges[recv_chunk_start].first,
        ranges[(recv_chunk_end - 1 + ranges.size()) % ranges.size()].second -
            ranges[recv_chunk_start].first);

    SendWait(to_rank);

    start_chunk = recv_chunk_start;
    end_chunk = recv_chunk_end;
  }

  // AllGather (recursive doubling)
  for (int step = 0; step < world_size_exp; step++) {
    int to_rank = my_rank_ ^ (1 << (world_size_exp - step - 1));

    int send_chunk_start, send_chunk_end, recv_chunk_start, recv_chunk_end;
    if (my_rank_ > to_rank) {
      // I send front chunk
      send_chunk_start = start_chunk;
      send_chunk_end = end_chunk;
      recv_chunk_start = send_chunk_end;
      recv_chunk_end = recv_chunk_start + end_chunk - start_chunk;
    } else {
      // I send back chunk
      send_chunk_start = start_chunk;
      send_chunk_end = end_chunk;
      recv_chunk_end = send_chunk_start;
      recv_chunk_start = recv_chunk_end - (end_chunk - start_chunk);
    }

    Send(to_rank, recvbuf + ranges[send_chunk_start].first,
         sizeof(T) *
             (ranges[(send_chunk_end - 1 + ranges.size()) % ranges.size()]
                  .second -
              ranges[send_chunk_start].first),
         false);

    Recv(to_rank, recvbuf + ranges[recv_chunk_start].first,
         sizeof(T) *
             (ranges[(recv_chunk_end - 1 + ranges.size()) % ranges.size()]
                  .second -
              ranges[recv_chunk_start].first),
         false);

    RecvWait(to_rank);
    SendWait(to_rank);

    start_chunk = std::min(send_chunk_start, recv_chunk_start);
    end_chunk = std::max(send_chunk_end, recv_chunk_end);
  }

  free(tmp_buffer);
}
