/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/pcx_allreduce_king.h"

namespace gloo {

int p2p_exchange(void *comm, volatile void *send_buf, volatile void *recv_buf,
                 size_t size, uint32_t peer, uint32_t tag) {

//  fprintf(stderr,"p2p_exchange called: size %d, peer %d, tag %d\n");


  std::shared_ptr<Context> *ctx = static_cast<std::shared_ptr<Context> *>(comm);
  auto &pair = (*ctx)->getPair(peer);
  auto sendBuf = pair->createSendBuffer(tag, (void *)send_buf, size);
  auto recvBuf = pair->createRecvBuffer(tag, (void *)recv_buf, size);

  sendBuf->send();
//  fprintf(stderr,"p2p_exchange sent\n");

  sendBuf->waitSend();

//  fprintf(stderr,"p2p_exchange wait sent done\n");

  recvBuf->waitRecv();

//  fprintf(stderr,"p2p_exchange wait recv done\n");

}

}