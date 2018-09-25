// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#include "ibcomm/ibverbs_communicator.h"

#include <sys/time.h>

#include <cassert>
#include <cerrno>
#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include "ibcomm/util.h"

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

IBVerbsCommunicator::IBVerbsCommunicator() {}
IBVerbsCommunicator::IBVerbsCommunicator(int world_size) { Init(world_size); }

void IBVerbsCommunicator::Init(int world_size) {
  if (initialized_) {
    util::IbcommWarning(__FILE__, __LINE__,
                        "IBVerbsCommunicator is already initialized.");
    return;
  }

  int ret = ibv_fork_init();
  if (ret) {
    int errno_backup = errno;
    util::IbcommWarning(__FILE__, __LINE__, "Failure: ibv_fork_init (errno=%d)",
                        errno_backup);
  }

  int devices;
  dev_list_ = ibv_get_device_list(&devices);

  if (!dev_list_) {
    int errno_backup = errno;
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_get_device_list (errno=%d)", errno_backup);
  }

  for (int i = 0; i < devices; i++) {
    ibv_device* device = dev_list_[i];

    if (!device) {
      continue;
    }

    context_ = ibv_open_device(device);

    if (!context_) {
      continue;
    }
  }

  if (!context_) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: No HCA can use");
  }

  ret = ibv_query_port(context_, 1, &port_attr_);

  if (ret != 0 || port_attr_.lid == 0) {
    // error handling
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_query_port");
  }

  pd_ = ibv_alloc_pd(context_);

  if (!pd_) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_alloc_pd");
  }

  world_size_ = world_size;
  pq_world_ = std::vector<ProcessQueue>(world_size_);
  psn_world_ = std::vector<uint32_t>(world_size_);
  mr_world_ = std::vector<std::pair<struct ibv_mr*, struct ibv_mr*>>(
      world_size_, std::pair<struct ibv_mr*, struct ibv_mr*>(NULL, NULL));

#ifdef USE_CUDA
  PrepareMemoryPool();
#endif

  initialized_ = true;
}

IBVerbsCommunicator::~IBVerbsCommunicator() {
  // release queues
  for (size_t i = 0; i < pq_world_.size(); i++) {
    if (pq_world_[i].queue_pair != NULL)
      ibv_destroy_qp(pq_world_[i].queue_pair);

    if (pq_world_[i].recv_complete_queue != NULL)
      ibv_destroy_cq(pq_world_[i].recv_complete_queue);

    if (pq_world_[i].send_complete_queue != NULL)
      ibv_destroy_cq(pq_world_[i].send_complete_queue);
  }

  // release memory region which is nonblocking-io but not freed.
  for (size_t i = 0; i < mr_world_.size(); i++) {
    // send
    if (mr_world_[i].first != NULL) {
      ibv_dereg_mr(mr_world_[i].first);
      mr_world_[i].first = NULL;
    }

    // recv
    if (mr_world_[i].second != NULL) {
      ibv_dereg_mr(mr_world_[i].second);
      mr_world_[i].second = NULL;
    }
  }

#ifdef USE_CUDA
  pool_.reset();
#endif

  if (pd_ != NULL) {
    ibv_dealloc_pd(pd_);
  }

  if (context_ != NULL) {
    ibv_close_device(context_);
  }

  if (dev_list_ != NULL) {
    ibv_free_device_list(dev_list_);
  }

#ifdef USE_TRACE
  DumpTrace();
#endif

#ifdef USE_CUDA
  if (tmp_gpu_buffer_ != NULL) {
    CUDACHECK(cudaFree(tmp_gpu_buffer_));
    tmp_gpu_buffer_ = NULL;
  }
#endif
}

namespace {
double timeDiffMillis(const struct timespec& t1, const struct timespec& t2) {
  return (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) * 1e-6;
}

void DumpTraceFromVector(std::ofstream& stream, struct timespec origin,
                         const std::vector<struct timespec>& vector) {
  for (int i = 0; i < vector.size(); i += 2) {
    stream << timeDiffMillis(origin, vector[i]) << ",";
    stream << timeDiffMillis(vector[i], vector[i + 1]) << ",";
  }
}
}  // namespace

void IBVerbsCommunicator::DumpTrace() const {
  std::stringstream ss;
  const char* base = getenv("IBCOMM_TRACE_FILE");
  base = base ? base : "ibcomm_trace";
  ss << base << "_" << my_rank_ << ".dat";
  std::ofstream trace_log;
  trace_log.open(ss.str().c_str());

  if (!trace_log.good()) {
    std::cerr << "ERROR: ofstream open failed" << std::endl;
  } else {
    trace_log << std::scientific;

    trace_log << "received,";
    DumpTraceFromVector(trace_log, trace_start_, trace_received_);
    trace_log << std::endl;

    trace_log << "reduced,";
    DumpTraceFromVector(trace_log, trace_start_, trace_reduced_);
    trace_log << std::endl;

    trace_log << "issue-send,";
    DumpTraceFromVector(trace_log, trace_start_, trace_issue_send_);
    trace_log << std::endl;

    trace_log << "issue-copy-kernel,";
    DumpTraceFromVector(trace_log, trace_start_, trace_issue_copy_kernel_);
    trace_log << std::endl;

    trace_log << "issue-redu-kernel,";
    DumpTraceFromVector(trace_log, trace_start_, trace_issue_redu_kernel_);
    trace_log << std::endl;

    trace_log << "issue-recv,";
    DumpTraceFromVector(trace_log, trace_start_, trace_issue_recv_);
    trace_log << std::endl;

    trace_log << "other,";
    DumpTraceFromVector(trace_log, trace_start_, trace_other_);
    trace_log << std::endl;
  }
}

void IBVerbsCommunicator::SetTimerBase() {
  clock_gettime(CLOCK_MONOTONIC_RAW, &trace_start_);
}

namespace {
void modify_qp(struct ibv_qp* qp, uint32_t src_psn, uint16_t dest_lid,
               uint32_t dest_pqn, uint32_t dest_psn) {
  int ret;

  struct ibv_qp_attr init_attr = {};
  init_attr.qp_state = IBV_QPS_INIT;
  init_attr.port_num = 1;
  init_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;

  ret = ibv_modify_qp(
      qp, &init_attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  if (ret != 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_modify_qp(1)");
  }

  struct ibv_qp_attr rtr_attr = {};
  rtr_attr.qp_state = IBV_QPS_RTR;
  rtr_attr.path_mtu = IBV_MTU_4096;
  rtr_attr.dest_qp_num = dest_pqn;
  rtr_attr.rq_psn = dest_psn;
  rtr_attr.max_dest_rd_atomic = 0;

  // retry_speed faster
  rtr_attr.min_rnr_timer = 1;

  // retry_speed slower
  // rtr_attr.min_rnr_timer = 0;

  rtr_attr.ah_attr.is_global = 0;
  rtr_attr.ah_attr.dlid = dest_lid;
  rtr_attr.ah_attr.sl = 0;
  rtr_attr.ah_attr.src_path_bits = 0;
  rtr_attr.ah_attr.port_num = 1;

  ret = ibv_modify_qp(qp, &rtr_attr,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                          IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_modify_qp(2)");
  }

  struct ibv_qp_attr rts_attr = {};
  rts_attr.qp_state = IBV_QPS_RTS;
  rts_attr.timeout = 1;
  rts_attr.retry_cnt = 7;
  rts_attr.rnr_retry = 7;
  rts_attr.sq_psn = src_psn;
  rts_attr.max_rd_atomic = 0;

  ret = ibv_modify_qp(qp, &rts_attr,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                          IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                          IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret != 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_modify_qp(3)");
  }
}
}  // namespace

struct ProcessInfo IBVerbsCommunicator::RegisterProcess(
    int dest_rank, struct ProcessInfo pinfo) {
  struct ProcessInfo my_pinfo = CreateQueuePair(dest_rank);

  modify_qp(pq_world_[dest_rank].queue_pair, psn_world_[dest_rank], pinfo.lid,
            pinfo.qp_n, pinfo.psn);

  return my_pinfo;
}

struct ProcessInfo IBVerbsCommunicator::CreateQueuePair(int dest_rank) {
  ibv_cq *send_complete_queue, *recv_complete_queue;

  send_complete_queue = ibv_create_cq(context_, 1024 * 1024, NULL, NULL, 0);
  recv_complete_queue = ibv_create_cq(context_, 1024 * 1024, NULL, NULL, 0);

  if (!send_complete_queue) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_create_cq of send cq");
  }

  if (!recv_complete_queue) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_create_cq of recv cq");
  }

  uint32_t my_psn = random() % 0xFFFFFF;
  psn_world_[dest_rank] = my_psn;

  struct ibv_qp_init_attr qp_init_attr = {};
  qp_init_attr.qp_type = IBV_QPT_RC;
  qp_init_attr.send_cq = send_complete_queue;
  qp_init_attr.recv_cq = recv_complete_queue;
  qp_init_attr.cap.max_send_wr = 8192;
  qp_init_attr.cap.max_recv_wr = 8192;
  qp_init_attr.cap.max_send_sge = 1;
  qp_init_attr.cap.max_recv_sge = 1;
  qp_init_attr.sq_sig_all = 1;

  struct ibv_qp* queue_pair;
  queue_pair = ibv_create_qp(pd_, &qp_init_attr);

  if (!queue_pair) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_create_cq");
  }

  pq_world_[dest_rank] =
      ProcessQueue(send_complete_queue, recv_complete_queue, queue_pair);

  struct ProcessInfo my_pinfo = {};
  my_pinfo.lid = port_attr_.lid;
  my_pinfo.psn = my_psn;
  my_pinfo.qp_n = queue_pair->qp_num;

  return my_pinfo;
}

void IBVerbsCommunicator::RegisterQueuePair(int dest_rank,
                                            struct ProcessInfo pinfo) {
  const auto& pqueue = pq_world_[dest_rank];

  modify_qp(pqueue.queue_pair, psn_world_[dest_rank], pinfo.lid, pinfo.qp_n,
            pinfo.psn);
}

void IBVerbsCommunicator::RegisterMyself(int my_rank) {
  this->my_rank_ = my_rank;
}

struct ibv_mr* IBVerbsCommunicator::RegisterSendBuf(const void* buf,
                                                    size_t len) {
  struct ibv_mr* mr_buf = ibv_reg_mr(pd_, const_cast<void*>(buf), len, 0);
  if (mr_buf == 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_reg_mr on send");
  }

  return mr_buf;
}

void IBVerbsCommunicator::Send(int dest_rank, const void* buf, size_t len,
                               bool blocking) {
  auto& save = mr_world_[dest_rank].first;

  if (save != NULL) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
                      "SendWait must be called before next non-blocking send.");
  }

  save = RegisterSendBuf(buf, len);

  SendRegistered(dest_rank, buf, save, len, blocking);
}

void IBVerbsCommunicator::SendRegistered(int dest_rank, const void* buf,
                                         struct ibv_mr* mr_buf, size_t len,
                                         bool blocking) {
  int ret;
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)(uintptr_t)buf;
  sge.length = len;
  sge.lkey = mr_buf->lkey;

  struct ibv_send_wr send_wr = {};
  send_wr.wr_id = (uint64_t)(uintptr_t)buf;
  send_wr.sg_list = &sge;
  send_wr.num_sge = 1;
  send_wr.opcode = IBV_WR_SEND;

  const auto& pq = pq_world_[dest_rank];

  struct ibv_send_wr* bad_wr;
  ret = ibv_post_send(pq.queue_pair, &send_wr, &bad_wr);
  if (ret != 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_post_send");
  }

  if (blocking) {
    SendWait(dest_rank);
  }
}

bool IBVerbsCommunicator::SendPoll(int dest_rank) {
  int ret;
  const auto& pq = pq_world_[dest_rank];
  struct ibv_wc wc = {};
  bool ok = false;

  ret = ibv_poll_cq(pq.send_complete_queue, 1, &wc);
  if (ret == 0) return false;

  if (ret < 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_poll_cq");
  }

  if (wc.status != IBV_WC_SUCCESS && wc.status == IBV_WC_LOC_PROT_ERR) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: send completion error %d", wc.status);
  }

  switch (wc.opcode) {
    case IBV_WC_SEND:
      ok = true;

      break;

    default:
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                        "Failure: SendPoll %d", wc.opcode);
  }

  if (ok) {
    // unregister memory region from non-blocking io wait list
    auto& save = mr_world_[dest_rank].first;
    if (save != NULL) {
      ibv_dereg_mr(save);
      save = NULL;
    }

    return true;
  }

  return false;
}

void IBVerbsCommunicator::SendWait(int dest_rank) {
  while (!SendPoll(dest_rank)) {
  }

  // unregister memory region from non-blocking io wait list
  auto& save = mr_world_[dest_rank].first;
  if (save != NULL) {
    ibv_dereg_mr(save);
    save = NULL;
  }
}

struct ibv_mr* IBVerbsCommunicator::RegisterRecvBuf(void* buf, size_t len) {
  struct ibv_mr* mr_buf = ibv_reg_mr(pd_, buf, len, IBV_ACCESS_LOCAL_WRITE);
  if (mr_buf == 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_reg_mr on recv");
  }

  return mr_buf;
}

void IBVerbsCommunicator::Recv(int src_rank, void* buf, size_t len,
                               bool blocking) {
  auto& save = mr_world_[src_rank].second;

  if (save != NULL) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
                      "RecvWait must be called before next non-blocking send.");
  }

  save = RegisterRecvBuf(buf, len);

  RecvRegistered(src_rank, buf, save, len, blocking);
}

void IBVerbsCommunicator::RecvRegistered(int src_rank, const void* buf,
                                         struct ibv_mr* mr_buf, size_t len,
                                         bool blocking) {
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)(uintptr_t)buf;
  sge.length = len;
  sge.lkey = mr_buf->lkey;

  struct ibv_recv_wr recv_wr = {};
  recv_wr.wr_id = (uint64_t)(uintptr_t)buf;
  recv_wr.sg_list = &sge;
  recv_wr.num_sge = 1;

  const auto& pq = pq_world_[src_rank];

  struct ibv_recv_wr* bad_wr;
  int ret = ibv_post_recv(pq.queue_pair, &recv_wr, &bad_wr);
  if (ret != 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_post_recv");
  }

  if (blocking) {
    RecvWait(src_rank);
  }
}

bool IBVerbsCommunicator::RecvPoll(int src_rank) {
  int ret;
  const auto& pq = pq_world_[src_rank];
  struct ibv_wc wc = {};
  bool ok = false;

  ret = ibv_poll_cq(pq.recv_complete_queue, 1, &wc);
  if (ret == 0) return false;

  if (ret < 0) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: ibv_poll_cq");
  }

  if (wc.status != IBV_WC_SUCCESS && wc.status == IBV_WC_LOC_PROT_ERR) {
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                      "Failure: recv completion error %d", wc.status);
  }

  switch (wc.opcode) {
    case IBV_WC_RECV:
      ok = true;

      break;

    default:
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::IBVERBS_ERROR,
                        "Failure: RecvPoll %d", wc.opcode);
  }

  if (ok) {
    // unregister memory region from non-blocking io wait list
    auto& save = mr_world_[src_rank].second;
    if (save != NULL) {
      ibv_dereg_mr(save);
      save = NULL;
    }
    return true;
  }
  return false;
}

void IBVerbsCommunicator::RecvWait(int src_rank) {
  while (!RecvPoll(src_rank)) {
  }

  // unregister memory region from non-blocking io wait list
  auto& save = mr_world_[src_rank].second;
  if (save != NULL) {
    ibv_dereg_mr(save);
    save = NULL;
  }
}

void IBVerbsCommunicator::Bcast(void* buf, size_t len, int root) {
  // This function provides naive Bcast;

  if (my_rank_ == root) {
    // Bcast root
    for (size_t i = 0; i < pq_world_.size(); i++) {
      if (static_cast<int>(i) == my_rank_) continue;

      Send(i, buf, len, false);
    }

    for (size_t i = 0; i < pq_world_.size(); i++) {
      if (static_cast<int>(i) == my_rank_) continue;

      SendWait(i);
    }
  } else {
    // Bcast non-root
    Recv(root, buf, len);
  }
}

void IBVerbsCommunicator::PopMrAndDereg(std::queue<struct ibv_mr*>* q) {
  if (q == nullptr)
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                      "q is nullptr.");

  ibv_dereg_mr(q->front());
  q->pop();
}

namespace {
int ReadChunkSize() {
  const char* size = getenv("IBCOMM_CHUNKSIZE");

  if (size != NULL) {
    int size_int = atoi(size);

    if (size_int <= 0)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "IBCOMM_CHUNKSIZE must be greater than 1");

    return size_int;
  }

  return -1;  // use default size
}
};  // namespace

std::vector<std::pair<size_t, size_t>> IBVerbsCommunicator::SplitBuffer(
    size_t len_elements, size_t sizeof_element) {
  int chunks;
  size_t elements_per_chunk;

  if (len_elements < world_size_) {
    util::IbcommError(
        __FILE__, __LINE__, util::IBCOMM_ERROR_CODE::NOT_SUPPORTED,
        "Input vector is too short for current Allreduce algorithm.\n"
        "Incrase the number of the input vector to be larger than number of "
        "processes.\n");
  }

  int env_chunk_bytes = ReadChunkSize();
  if (env_chunk_bytes != -1) {
    // chunk_size is selected manually.
    if (env_chunk_bytes % sizeof_element != 0) {
      util::IbcommError(
          __FILE__, __LINE__, util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
          "Selected `IBCOMM_CHUNKSIZE` is not divisible by `sizeof(T)`.");
    }

    elements_per_chunk = env_chunk_bytes / sizeof_element;
    chunks = util::ceilDiv((size_t)len_elements, (size_t)elements_per_chunk);

    if (chunks < 2 * world_size_) {
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "Selected `IBCOMM_CHUNKSIZE` is too large.\n"
                        "Satisfy 2 * `world_size` <= `allreduce_bufsize` / "
                        "`IBCOMM_CHUNKSIZE`.");
    }
  } else {
    chunks = 4 * world_size_;
    elements_per_chunk = util::ceilDiv((size_t)len_elements, (size_t)chunks);
  }

  std::vector<std::pair<size_t, size_t>> ranges;
  for (auto i = 0; i < chunks; i++) {
    int start_index = elements_per_chunk * i;
    int end_index = std::min(len_elements, elements_per_chunk * (i + 1));

    if (start_index < end_index) ranges.emplace_back(start_index, end_index);
  }

  return ranges;
}

std::vector<std::vector<int>> IBVerbsCommunicator::GetRankToChunk(
    const std::vector<std::pair<size_t, size_t>>& ranges) {
  std::vector<std::vector<int>> rank_to_chunk(world_size_);
  size_t chunks_per_rank = ranges.size() / world_size_;
  size_t chunks_per_rank_remainer = ranges.size() % world_size_;
  int chunk_id = 0;

  for (int i = 0; i < world_size_; i++) {
    for (int j = 0; j < chunks_per_rank; j++) {
      rank_to_chunk[i].push_back(chunk_id);
      chunk_id++;
    }

    if (i < chunks_per_rank_remainer) {
      rank_to_chunk[i].push_back(chunk_id);
      chunk_id++;
    }
  }

  assert(chunk_id == ranges.size());

  return rank_to_chunk;
}
