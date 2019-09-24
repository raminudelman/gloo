/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/pcx_allreduce_ring.h"

/**
send_buf_left - buffer to send to left
send_buf_right - buffer to send to right
recv_buf_left - buffer to recieve from left peer
recv_buf_right - buffer to recieve from right peer
size           - all buffers have the same size
peer_left      - left rank
perr_right     - right rank
**/
int ring_exchange(void *comm, volatile void *send_buf_left, volatile void *send_buf_right, volatile void *recv_buf_left,
                  volatile void *recv_buf_right, size_t size, uint32_t peer_left, uint32_t peer_right, uint32_t tag1, uint32_t tag2) {
  std::shared_ptr<gloo::Context> *ctx = static_cast<std::shared_ptr<gloo::Context> *>(comm);
  //peer to the left
  auto &pair_left = (*ctx)->getPair(peer_left);
  //peer to the right
  auto &pair_right = (*ctx)->getPair(peer_right);

  //buffer that we will send to left rank
  auto send_buf_left_p = pair_left->createSendBuffer(tag1, (void *)send_buf_left, size);

  //buffer that we will send to right rank
  auto send_buf_right_p = pair_right->createSendBuffer(tag2, (void *)send_buf_right, size);

  auto recv_buf_left_p = pair_left->createRecvBuffer(tag2, (void *)recv_buf_left, size);
  auto recv_buf_right_p = pair_right->createRecvBuffer(tag1, (void *)recv_buf_right, size);

  send_buf_left_p->send();
  send_buf_left_p->waitSend();
  recv_buf_right_p->waitRecv();

  send_buf_right_p->send();
  send_buf_right_p->waitSend();
  recv_buf_left_p->waitRecv();
}
