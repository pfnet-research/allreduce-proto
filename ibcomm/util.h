// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#pragma once

#include <sys/time.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef USE_CUDA
#define CUDACHECK(cmd)                                       \
  do {                                                       \
    cudaError_t e = cmd;                                     \
    if (e != cudaSuccess) {                                  \
      util::IbcommError(__FILE__, __LINE__,                  \
                        util::IBCOMM_ERROR_CODE::CUDA_ERROR, \
                        cudaGetErrorString(e));              \
    }                                                        \
                                                             \
  } while (0)
#endif

namespace util {
enum class IBCOMM_ERROR_CODE : int {
  INVALID_ARGUMENT = 1,

  // Error occured in InfiniBand Verbs call.
  IBVERBS_ERROR = 2,

  // Error occured in CUDA call.
  CUDA_ERROR = 3,

  NOT_SUPPORTED = 4
};

template <typename... Args>
void IbcommError(const char* filename, int line, IBCOMM_ERROR_CODE error_code,
                 const char* format, Args const&... args) {
  fprintf(stderr, "Error occured at %s:L%d.\n", filename, line);
  fprintf(stderr, format, args...);
  fputs("", stderr);

  exit(static_cast<int>(error_code));
}

template <typename... Args>
void IbcommWarning(const char* filename, int line, const char* format,
                   Args const&... args) {
  fprintf(stderr, "Warning occured at %s:L%d.\n", filename, line);
  fprintf(stderr, format, args...);
  fputs("", stderr);
}

inline void trace(std::vector<struct timespec>* v) {
  if (v == nullptr)
    util::IbcommError(__FILE__, __LINE__,
                      util::IBCOMM_ERROR_CODE::INVALID_ARGUMENT,
                      "v is nullptr.");
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  v->push_back(ts);
}

class MalformedNumber : public std::runtime_error {
 public:
  explicit MalformedNumber(const std::string& ss)
      : std::runtime_error(ss.c_str()) {}
};

// Parse a string and get a buffer size or chunk size. SI prefix is supported.
// If any error occurs, success is set to false and the error message is
// assigned to msg.
inline int64_t parse_number(const char* str) {
  std::string n;
  int64_t multiply = 1;
  int pos = 0;
  const int len = strlen(str);

  if (str[0] == '+' || str[0] == '-') {
    // accept '-' for now to detect value range error.
    n += str[0];
    pos = 1;
  }

  while (isdigit(str[pos]) && pos < len) {
    n += str[pos];
    pos++;
  }

  if (n.size() == 0) {
    // there seems no number
    std::stringstream ss;
    ss << "Illegal number format prefix in '" << str << "'";
    throw MalformedNumber(ss.str());
  }

  if (pos < len) {
    // parse SI prefix
    switch (str[pos]) {
      case 'k':
      case 'K':
        multiply = 1024ul;
        pos++;
        break;
      case 'm':
      case 'M':
        multiply = 1024ul * 1024;
        pos++;
        break;
      case 'g':
      case 'G':
        multiply = 1024ul * 1024 * 1024;
        pos++;
        break;
        // default:
        //   {
        //     std::stringstream ss;
        //     ss << "Illegal SI prefix in '" << str << "'";
        //     throw MalformedNumber(ss.str());
        //   }
    }
  }

  if (pos < len) {
    // Last 'b' or 'B' (bytes) is optional. Other characters are not allowed.
    if (!(str[pos] == 'b' || str[pos] == 'B')) {
      std::stringstream ss;
      ss << "Illegal SI prefix in '" << str << "'";
      throw MalformedNumber(ss.str());
    }
    pos++;
  }
  if (pos < len) {
    std::stringstream ss;
    ss << "Illegal number format prefix in '" << str << "'";
    throw MalformedNumber(ss.str());
  }

  int64_t n2 = atol(n.c_str());
  return n2 * multiply;
}

template <typename T>
inline T ceilDiv(T v1, T v2) {
  return v1 % v2 ? v1 / v2 + 1 : v1 / v2;
}

inline int GetExpOfTwo(int n) {
  int p = 0;

  while (n != 0) {
    if (n % 2 == 1) {
      if (n == 1)
        return p;
      else
        return 0;
    }

    p++;

    n >>= 1;
  }

  return 0;

  // returns p (2^p == n)
}
};  // namespace util
