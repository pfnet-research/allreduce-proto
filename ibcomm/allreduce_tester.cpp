// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.
// allreduce_tester.cpp
//
// A helper program for allreduce integration test.
//
// Usage:
//   $ mpiexec -n ${NP} ./allreduce_tester [algorithm] [buffer size] [init expr]
//   [check expr]
//
//      * [algorithm]    : Name of Allreduce Algorithm. "ring" and
//      "rabenseifner" is supported.
//
//      * [buffer size]  : Size of the target buffer. Suffix "k", "m", "g" are
//      allowed.
//                         ex.) 1024, 128M, 10k
//      * [init expr]    : Target bufffer is initialized with this expression in
//      an elementwise manner.
//                         For details of expressions, see
//                         https://github.com/codeplea/tinyexpr Additional
//                         variables are supported:
//                           - p  : Process rank
//                           - np : Number of processes (e.g. mpi_size)
//                           - n  : Number of elementso of the target buffer
//                           (NOT size in bytes)
//                           - nb : Size of the target buffer in bytes
//                           - i  : Index of the element in buffer
//                         ex.)
//                           (1) init_expr = "1", n = 4, np = 2
//                             Rank 0:  [1, 1, 1, 1]
//                             Rank 1:  [1, 1, 1, 1]
//                           (2) init_expr = "1/np*p+i", n = 4, np = 2
//                             Rank 0: [0.0, 1.0, 2.0, 3.0]   # [1/2*0+0,
//                             1/2*0+1, ...] Rank 1: [0.5, 1.5, 2.5, 3.5]   #
//                             [1/2*1+0, 1/2*1+1, ... ]
//
//      * [check expr]   : Target buffer is checked after Allreduce operation
//      using check expr.
//                         The grammar of expressions is identical to [init
//                         expr]
//

#include <cuda_runtime.h>
#include <mpi.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include "grumpi/grumpi.hpp"
#include "ibcomm/ibverbs_communicator.h"
#include "ibcomm/util.h"
#include "tinyexpr/tinyexpr.h"

class TinyExpr {
  std::vector<te_variable> vars_;
  te_expr *expr_;

 public:
  TinyExpr() : vars_(), expr_(nullptr) {}

  void set_variable(const char *name, const void *address, int type,
                    void *context) {
    te_variable va;
    va.name = name;
    va.address = address;
    va.type = type;
    va.context = context;
    vars_.push_back(va);
  }

  void compile(const std::string &expr) { compile(expr.c_str()); }

  void compile(const char *expr) {
    int err;
    expr_ =
        te_compile(expr, vars_.data(), static_cast<int>(vars_.size()), &err);

    if (!expr_) {
      std::stringstream ss;
      ss << "Invalid expression: '" << expr << "'";
      throw std::runtime_error(ss.str());
    }
  }

  double eval() {
    if (!expr_) {
      throw std::runtime_error("Expression must be compiled before eval()");
    }
    return te_eval(expr_);
  }
};

class Communicator {
  MPI_Comm mpi_comm_;
  int size_;
  int rank_;
  std::unique_ptr<IBVerbsCommunicator> ibcomm_;

 public:
  explicit Communicator(MPI_Comm comm = MPI_COMM_WORLD) : mpi_comm_(comm) {
    MPI_Comm_rank(mpi_comm_, &rank_);
    MPI_Comm_size(mpi_comm_, &size_);
    ibcomm_.reset(new IBVerbsCommunicator(size_));

    std::vector<uint32_t> qps(size_ * 3);

    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        continue;
      }
      ProcessInfo pinfo = ibcomm_->CreateQueuePair(i);
      qps[i * 3 + 0] = pinfo.lid;
      qps[i * 3 + 1] = pinfo.qp_n;
      qps[i * 3 + 2] = pinfo.psn;
    }

    MPI_Alltoall(MPI_IN_PLACE, 3, MPI_UINT32_T, qps.data(), 3, MPI_UINT32_T,
                 comm);

    for (int i = 0; i < size_; i++) {
      if (i == rank_) {
        ibcomm_->RegisterMyself(i);
      } else {
        ProcessInfo pinfo;
        pinfo.lid = qps[i * 3 + 0];
        pinfo.qp_n = qps[i * 3 + 1];
        pinfo.psn = qps[i * 3 + 2];
        ibcomm_->RegisterQueuePair(i, pinfo);
      }
    }
  }

  void die(const std::string &errmsg, int retcode = 1) {
    if (rank_ == 0) {
      std::cerr << errmsg << std::endl;
    }
    exit(retcode);
  }

  template <class T>
  void allreduce(const std::string &algorithm_type,
                 const thrust::device_vector<T> &sendbuf_d,
                 thrust::device_vector<T> *recvbuf_d) {
    if (recvbuf_d == nullptr)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "recvbuf_d is nullptr.");

    if (algorithm_type == "ring") {
      ibcomm_->AllreduceRingCuda(sendbuf_d.data().get(),
                                 recvbuf_d->data().get(), sendbuf_d.size());
    } else if (algorithm_type == "rabenseifner") {
      ibcomm_->AllreduceRabenseifnerCuda(
          sendbuf_d.data().get(), recvbuf_d->data().get(), sendbuf_d.size());
    } else {
      die("Error: Unsupported algorithm");
    }
  }
};

/**
 * Application main class
 */
template <class T>
class AllreduceTester {
 public:
  using ElemType = T;

 private:
  MPI_Comm comm_;
  int mpi_rank_;
  int mpi_size_;

  Communicator ibcomm_;

  // target array size
  size_t array_nbytes_;  // array size in bytes
  size_t num_elems_;     // array length

  double var_p;   // "p"  variable in expressions (process rank)
  double var_np;  // "np" variable in expressions (number of processes, i.e.
                  // mpi_size)
  double var_n;   // "n"  variable in expressions (number of elements in buffer)
  double var_nb;  // "nb" variable in expressions (size of buffer in bytes)

  std::unique_ptr<TinyExpr> init_expr_;
  std::unique_ptr<TinyExpr> check_expr_;

  void usage(int argc, char **argv) {
    if (mpi_rank_ == 0) {
      std::cerr << "Usage: " << argv[0] << " "
                << "[algorithm] "
                << "[array size (Bytes)] "
                << "[init expr] "
                << "[check expr]" << std::endl;
    }
    exit(-1);
  }

  void die(const std::string &errmsg, int retcode = 1) {
    if (mpi_rank_ == 0) {
      std::cerr << errmsg << std::endl;
    }
    exit(retcode);
  }

  size_t parse_nbytes(const char *src) {
    size_t i = 0;
    size_t n = 0;

    while (std::isdigit(src[i])) {
      n = n * 10 + (src[i] - '0');
      i++;
    }

    if (src[i] != 0) {
      switch (src[i]) {
        case 'k':
        case 'K':
          n *= 1024;
          break;
        case 'm':
        case 'M':
          n *= 1024 * 1024;
          break;
        case 'g':
        case 'G':
          n *= 1024 * 1024 * 1024;
          break;
        default:
          std::stringstream ss;
          ss << "Cannot parse an array size: '" << src << "'" << std::endl;
          die(ss.str());
      }
    }

    i++;

    return n;
  }

  std::tuple<std::string, size_t, std::string, std::string> parse_args(
      int argc, char **argv) {
    if (argc != 5) {
      usage(argc, argv);
    }
    // Parse argument 1
    std::string algorithm = argv[1];

    // Parse argument 2 (array length (bytes))
    size_t nbytes = parse_nbytes(argv[2]);

    // Parse argument 3 (initializing expression)
    std::string init_expr = argv[3];

    std::string check_expr = argv[4];

    return std::make_tuple(algorithm, nbytes, init_expr, check_expr);
  }

  void setup_sendbuf(thrust::host_vector<T> *buf,
                     const std::string &init_expr_str) {
    if (buf == nullptr)
      util::IbcommError(__FILE__, __LINE__,
                        util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                        "buf is nullptr.");

    TinyExpr expr;

    expr.set_variable("p", &var_p, TE_VARIABLE, nullptr);
    expr.set_variable("np", &var_np, TE_VARIABLE, nullptr);
    expr.set_variable("n", &var_n, TE_VARIABLE, nullptr);
    expr.set_variable("nb", &var_nb, TE_VARIABLE, nullptr);

    for (size_t i = 0; i < buf->size(); i++) {
      double var_i = i;
      expr.set_variable("i", &var_i, TE_VARIABLE, nullptr);
      expr.compile(init_expr_str);
      (*buf)[i] = expr.eval();
    }
  }

  std::tuple<bool, std::vector<std::string>> check_recvbuf(
      const thrust::host_vector<T> &buf, const std::string &check_expr_str) {
    constexpr double eps = 1e-12;

    std::vector<std::string> msgs;
    bool check_ok = true;

    TinyExpr expr;

    expr.set_variable("p", &var_p, TE_VARIABLE, nullptr);
    expr.set_variable("np", &var_np, TE_VARIABLE, nullptr);
    expr.set_variable("n", &var_n, TE_VARIABLE, nullptr);
    expr.set_variable("nb", &var_nb, TE_VARIABLE, nullptr);

    for (size_t i = 0; i < buf.size(); i++) {
      double var_i = i;
      expr.set_variable("i", &var_i, TE_VARIABLE, nullptr);
      expr.compile(check_expr_str);

      double res = buf[i];
      double ans = expr.eval();

      if (std::abs(res - ans) > eps) {
        std::stringstream ss;
        ss << "Error: Element [" << i << "] must be " << ans
           << " but actually is " << res;
        msgs.push_back(ss.str());
        check_ok = false;
      }
    }

    return std::make_tuple(check_ok, msgs);
  }

  void report_errors(bool check_ok, const std::vector<std::string> &msgs) {
    for (int i = 0; i < mpi_size_; i++) {
      if (i == mpi_rank_) {
        if (!check_ok) {
          size_t report_num = std::min(msgs.size(), (size_t)1000);
          for (size_t i = 0; i < report_num; i++) {
            std::cerr << "[Rank " << mpi_rank_ << "] " << msgs[i] << std::endl;
          }
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  thrust::host_vector<ElemType> run_allreduce(
      const std::string &algorithm_type,
      const thrust::host_vector<ElemType> &sendbuf) {
    thrust::host_vector<ElemType> recvbuf(num_elems_);
    thrust::device_vector<ElemType> recvbuf_d(num_elems_);
    thrust::device_vector<ElemType> sendbuf_d(num_elems_);

    sendbuf_d = sendbuf;

    ibcomm_.allreduce(algorithm_type, sendbuf_d, &recvbuf_d);

    recvbuf = recvbuf_d;
    return recvbuf;
  }

 public:
  explicit AllreduceTester(MPI_Comm comm) : comm_(comm), ibcomm_(comm_) {
    MPI_Comm_size(comm_, &mpi_size_);
    MPI_Comm_rank(comm_, &mpi_rank_);
  }

  // Check allreduce
  int run(int argc, char **argv) {
    std::string algorithm_type, init_expr_str, check_expr_str;
    std::tie(algorithm_type, array_nbytes_, init_expr_str, check_expr_str) =
        parse_args(argc, argv);

    std::vector<std::string> supported_algorithms = {"ring", "rabenseifner"};

    if (std::find(supported_algorithms.begin(), supported_algorithms.end(),
                  algorithm_type) == supported_algorithms.end()) {
      std::stringstream ss;
      ss << "Error: Unsupported algorithm " << algorithm_type << std::endl;

      ss << "Supported algorithms: ";
      for (auto algo : supported_algorithms) {
        ss << algo << ", ";
      }
      die(ss.str());
    }

    if (array_nbytes_ < sizeof(ElemType)) {
      if (mpi_rank_ == 0) {
        std::cerr << "Warning: specified array size is "
                  << "smaller than the element size(" << sizeof(ElemType)
                  << "). "
                  << "Ceiling it up to " << sizeof(ElemType) << " [bytes]"
                  << std::endl;
      }
      array_nbytes_ = sizeof(ElemType);
    }

    num_elems_ = array_nbytes_ / sizeof(ElemType);
    var_np = mpi_size_;
    var_p = mpi_rank_;
    var_n = array_nbytes_ / sizeof(ElemType);
    var_nb = array_nbytes_;

    thrust::host_vector<ElemType> sendbuf(num_elems_);
    setup_sendbuf(&sendbuf, init_expr_str);

    auto recvbuf = run_allreduce(algorithm_type, sendbuf);

    bool check_ok;
    std::vector<std::string> msgs;
    std::tie(check_ok, msgs) = check_recvbuf(recvbuf, check_expr_str);

    report_errors(check_ok, msgs);

    int status_all = (check_ok ? 0 : 1);
    MPI_Allreduce(&status_all, &status_all, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    return status_all;
  }
};

int main(int argc, char **argv) {
  using ElemType = int;

  MPI_Init(&argc, &argv);

  int ngpus = -1;
  CUDACHECK(cudaGetDeviceCount(&ngpus));

  int intra_rank;
  grumpi::Comm_local_rank(MPI_COMM_WORLD, &intra_rank);

  CUDACHECK(cudaSetDevice(intra_rank % ngpus));

  AllreduceTester<ElemType> tester(MPI_COMM_WORLD);

  int status = tester.run(argc, argv);

  MPI_Finalize();

  return status;
}
