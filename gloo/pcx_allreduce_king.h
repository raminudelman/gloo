/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#define PIPELINE_DEPTH 1

#include <alloca.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/context.h"
#include "gloo/pcx_allreduce_common.h"

#include <ctime>
#include <vector>

#ifdef DEBUG
#define PCX_KING_PRINT(args...) fprintf(stderr, "(%s: %d) in function %s [%d]: " \
                       ,__FILE__,__LINE__,__func__, contextRank_); fprintf(stderr, args)
#else
#define PCX_KING_PRINT(args...)
#endif

namespace gloo {

typedef struct mem_registration {
  Iov usr_vec;
  UmrMem *umr_mem;
  PipeMem *tmpMem;
} mem_registration_t;

int p2p_exchange(void *comm, volatile void *send_buf, volatile void *recv_buf,
                 size_t size, uint32_t peer, uint32_t tag);

class rd_peer_t {
public:
  rd_peer_t() : outgoing_buf(NULL), incoming_buf(NULL){};
  ~rd_peer_t() {
    delete (qp);
    delete (this->incoming_buf);
    delete (this->outgoing_buf);
  };

  DoublingQp *qp;
  NetMem *outgoing_buf;
  NetMem *incoming_buf;
};

typedef struct rd_connections {
  NetMem *result;

  CommGraph *graph;
  ManagementQp *mqp; // mqp stands for "Management Queue Pair"
  LoopbackQp *lqp;

  unsigned peers_cnt;
  rd_peer_t *peers; // Pointer to array of rd_peer_t
} rd_connections_t;

template <typename T> class PcxAllreduceKing : public Algorithm {
public:
  PcxAllreduceKing(const std::shared_ptr<Context> &context,
               const std::vector<T *> &ptrs, const int count,
               const ReductionFunction<T> *fn = ReductionFunction<T>::sum)
      : Algorithm(context), ptrs_(ptrs), count_(count),
        bytes_(count_ * sizeof(T)), fn_(fn) {
    if (this->contextSize_ == 1) {
      return;
    }

    // PCX performs the elements reduction on the NIC using Vector-CALC.
    // The reduction is on the number of elements in ptrs and another element
    // that is the result from a peer rank
    if ((ptrs.size() + 1) > MAX_LOCAL_VECTOR_SIZE_TO_REDUCE) {
      fprintf(stderr, "PCX does not support more than %d to be reduced on the NIC", MAX_LOCAL_VECTOR_SIZE_TO_REDUCE);
    }

    /* Step #1: Initialize verbs for all to use */
    PCX_KING_PRINT("starting PcxAllreduceKing \n");
    ibv_ = VerbCtx::getInstance();
    PCX_KING_PRINT("Verbs initiated \n");
    /* Step #2&3: Connect to the (recursive-doubling) peers and pre-post
     * operations */
    connect_and_prepare();
    mone = 0;
  }

  virtual ~PcxAllreduceKing() {
    teardown();
    deregister_memory();
    VerbCtx::remInstance();
  }

  void run() {
    PCX_KING_PRINT("King allreduce started \n");
    debug_write_input();
    rd_.graph->mqp->qp->db();

    PCX_KING_PRINT("Sent Doorbell to Management QP \n");

    rd_.graph->mqp->qp->rearm();

    PCX_KING_PRINT("Sent Rearm command to Management QP \n");
    int res = 0;
    uint64_t count = 0;
    while (res == 0) {
      res = rd_.lqp->qp->poll();
      ++count;
      if (contextRank_ == 0) {
        debug_hang_report(count);
      }
    }
    debug_check_output();
    ++mone;
    PCX_KING_PRINT("[%d] Done running PcxRingAllReduce \n", contextRank_);
  }

  void register_memory() {
    PRINT("locking... ");
    // std::lock_guard<std::mutex> lock(ibv_->m_);
    unsigned step_idx, step_count = 0;
    while ((1 << ++step_count) < contextSize_)
      ;

    pipeline = PIPELINE_DEPTH;
    while (step_count % pipeline) {
      --pipeline;
    }

    PRINT("Registering usr- memory... ");
    /* Register the user's buffers */
    for (int buf_idx = 0; buf_idx < ptrs_.size(); buf_idx++) {
      mem_.usr_vec.push_back(new UsrMem(ptrs_[buf_idx], bytes_, ibv_));
    }
    PRINT("UMR start... ");
    mem_.umr_mem = new UmrMem(mem_.usr_vec, ibv_);
    PRINT("UMR success");

    int mem_type = PCX_MEMORY_TYPE_MEMIC;
    mem_type = PCX_MEMORY_TYPE_HOST;

    mem_.tmpMem = new PipeMem(bytes_, pipeline, ibv_, mem_type);
  }

  void deregister_memory() {

    delete mem_.tmpMem;

    PRINT("Freeing UMR");
    int buf_idx;
    delete (mem_.umr_mem);
    PRINT("Freeing user memory");
    freeIov(mem_.usr_vec);
  }

  void connect_and_prepare() {
    int inputs = ptrs_.size();
    unsigned step_idx, step_count = 0;
    while ((1 << ++step_count) < contextSize_)
      ;

    PCX_KING_PRINT("step_count=%d \n", step_count);

    VerbCtx *ctx = (this->ibv_);
    // std::lock_guard<std::mutex> lock(ctx->m_);

    PCX_KING_PRINT("Locking the IB verbs context mtx \n");
    this->ibv_->mtx.lock();

    rd_.graph = new CommGraph(ctx);
    CommGraph *sess = rd_.graph;

    // Create a single management QP
    rd_.mqp = new ManagementQp(this->ibv_);
    sess->regQp(rd_.mqp);
    
    PCX_KING_PRINT("Created Management QP \n");

    /* Step #2: Register existing memory buffers with UMR */
    register_memory();

    /* Create a loopback QP */
    rd_.lqp = new LoopbackQp(this->ibv_);
    sess->regQp(rd_.lqp);
    LoopbackQp *lqp = rd_.lqp;
    PCX_KING_PRINT("Created Loopback QP \n");

    rd_.result = new HostMem(bytes_, ibv_);

    rd_.peers_cnt = step_count;
    rd_.peers = new rd_peer_t[step_count];
    if (!rd_.peers) {
      throw "malloc failed";
    }
    /* Establish a connection with each peer */

    for (step_idx = 0; step_idx < step_count; step_idx++) {
      /* calculate the rank of each peer */
      int leap = 1 << step_idx;
      if ((contextRank_ % (leap << 1)) >= leap) {
        leap *= -1;
      }
      uint32_t mypeer = contextRank_ + leap;

      uint32_t slot = this->context_->nextSlot();

      rd_.peers[step_idx].incoming_buf = new RefMem(mem_.tmpMem->next());

      rd_.peers[step_idx].qp =
          new DoublingQp(this->ibv_, &p2p_exchange, (void *)&(this->context_), mypeer,
                         slot, rd_.peers[step_idx].incoming_buf);
      sess->regQp(rd_.peers[step_idx].qp);                         
      PRINT("Creating RC QP - Done");
      Iov umr_iov{rd_.result, rd_.peers[step_idx].incoming_buf};
      rd_.peers[step_idx].outgoing_buf = new UmrMem(umr_iov, ibv_);
    }
    sess->reduce_write(lqp, mem_.umr_mem, rd_.result, inputs,
                      MLX5DV_VECTOR_CALC_OP_ADD,
                      MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32, false);
    sess->wait(lqp, false);
    for (step_idx = 0; step_idx < step_count; step_idx++) {
      if (step_idx >= pipeline) {
        sess->wait(rd_.peers[step_idx].qp, false); // Wait to receive credits.
      }
      // Send the data from "result" buffer to the peer and wait for the peer
      // to also send his data to this rank's buffer.
      sess->write(rd_.peers[step_idx].qp, rd_.result, true); // Send (write) the data to peer's buffer and requsting a completion for the sending side.
      sess->wait(rd_.peers[step_idx].qp, false); // Wait for the peer to send his data to this rank // TODO: First need to wait for the send and then need to wait for the receive. Or maybe even better, because of symmetry, in case this ranks sends data to the peer, the peer also sends it data to this rank, so no need to use "write with completion", and no need to wait for the "send" to complete.
      sess->wait(rd_.peers[step_idx].qp, true); // Wait for this rank to finish sending it's data to the peer.

      sess->reduce_write(lqp, rd_.peers[step_idx].outgoing_buf, rd_.result, 2,
                        MLX5DV_VECTOR_CALC_OP_ADD,
                        MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32, false);
      sess->wait(lqp, false);
      sess->send_credit(rd_.peers[(step_idx + pipeline) % step_count].qp);
    }
    for (uint32_t buf_idx = 0; buf_idx < inputs; buf_idx++) {
      sess->write(lqp, rd_.result, mem_.usr_vec[buf_idx], false);
    }
    sess->wait(lqp, false);
    for (step_idx = 0; step_idx < pipeline; step_idx++) {
      sess->wait(rd_.peers[step_idx].qp, false);
    }
    PRINT("Graph building - Done");
    rd_.graph->finish();

    this->ibv_->mtx.unlock();

    PCX_KING_PRINT("connect_and_prepare DONE \n");
  }

  void teardown() {
    delete (rd_.mqp);
    delete (rd_.lqp);
    delete (rd_.graph);
    delete (rd_.result);
    delete[](rd_.peers);
    PRINT("Teardown completed");
  }

  void debug_write_input() {
#ifdef VALIDITY_CHECK
    for (int i = 0; i < ptrs_.size(); ++i) {
      // fprintf(stderr, "Input %d:\n",i);
      float *buf = (float *)ptrs_[i];
      for (int k = 0; k < count_; ++k) {
        buf[k] = ((float)k + i) + contextRank_ + mone;
      }
      // print_values(buf, count_);
    }
#endif
  }

  void debug_hang_report(uint64_t &count) {
#ifdef HANG_REPORT

    unsigned step_count = 0;
    while ((1 << ++step_count) < contextSize_)
      ;

    if (count == 1000000000) {
      fprintf(stderr, "iteration: %d\n", mone);
      fprintf(stderr, "poll cnt: %d\n", rd_.lqp->qp->get_poll_cnt());
      fprintf(stderr, "management qp: \n");
      rd_.graph->mqp->print();
      fprintf(stderr, "loopback qp: \n");
      rd_.lqp->print();
      for (int k = 0; k < step_count; ++k) {
        fprintf(stderr, "rc qp %d: \n", k);
        rd_.peers[k].qp->print();
      }
      fprintf(stderr, "\n\n\n");
    }

#endif
  }

  void debug_check_output() {
#ifdef VALIDITY_CHECK

    unsigned step_count = 0;
    while ((1 << ++step_count) < contextSize_)
      ;

    for (int i = 0; i < ptrs_.size(); ++i) {
      // fprintf(stderr, "Output %d:\n",i);
      int err = 0;
      float *buf = (float *)ptrs_[i];
      // print_values(buf, count_);
      for (int k = 0; k < count_; ++k) {
        int expected_base =
            ((k + mone) * 2 + ptrs_.size() - 1) * ptrs_.size() / 2;
        int expected_max =
            ((k + mone + contextSize_ - 1) * 2 + ptrs_.size() - 1) *
            ptrs_.size() / 2;
        float expected_result =
            (float)(expected_base + expected_max) * contextSize_ / 2;
        float result = buf[k];
        if (result != expected_result) {
          fprintf(stderr,
                  "ERROR: In Iteration %d\n expected: %.2f, got: %.2f\n", mone,
                  expected_result, result);
          for (int i = 0; i < ptrs_.size(); ++i) {
            fprintf(stderr, "Input %d:\n", i);
            float buf[count_];
            for (int k = 0; k < count_; ++k) {
              buf[k] = ((float)k + i) + contextRank_ + mone;
            }
            print_values(buf, count_);
          }
          for (int i = 0; i < step_count; ++i) {
            fprintf(stderr, "Incoming %d:\n", i);
            float *buf =
                (float *)((void *)rd_.peers[i].incoming_buf->sg()->addr);
            print_values(buf, count_);
          }
          fprintf(stderr, "Output %d:\n", i);
          print_values(buf, count_);
          // err = 1;
          break;
        }
      }
      if (err) {
        break;
      }
    }
#endif
  }

protected:
  std::vector<T *> ptrs_;
  const int count_;
  const int bytes_;
  VerbCtx *ibv_;
  mem_registration_t mem_;
  rd_connections_t rd_;
  const ReductionFunction<T> *fn_;
  int mone;
  int pipeline;
};

} // namespace gloo
