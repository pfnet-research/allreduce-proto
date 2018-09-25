// Copyright (C) 2017-2018 by Preferred Networks, Inc. All right reserved.

#include <cassert>
#include <cerrno>
#include <cstdio>
#include "ibcomm/ibverbs_communicator.h"

int main(void) {
  IBVerbsCommunicator comm(2);

  ProcessInfo pinfoA = comm.CreateQueuePair(1);
  ProcessInfo pinfoB = comm.RegisterProcess(0, pinfoA);
  comm.RegisterQueuePair(1, pinfoB);

  int value = 10;
  int value2 = -1;

  comm.Send(0, &value, sizeof(value), false);
  comm.Recv(1, &value2, sizeof(value2));
  comm.SendWait(0);

  assert(value == value2);

  value2 = -1;
  comm.Send(0, &value2, sizeof(value), false);
  comm.Recv(1, &value, sizeof(value2));
  comm.SendWait(0);

  assert(value == value2);

  return 0;
}
