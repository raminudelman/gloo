/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found inpqp* LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#define RING_PIPELINE_DEPTH 1

#include <alloca.h>
#include <stddef.h>
#include <string.h>

#include "gloo/algorithm.h"
#include "gloo/context.h"
#include "third-party/pcx/pcx_mem.h"
#include "third-party/pcx/qps.h"

#include <ctime>
#include <vector>

#ifdef DEBUG
#define PCX_RING_PRINT(x) fprintf(stderr, "%s\n", x);
#else
#define PCX_RING_PRINT(x)
#endif

namespace gloo {

typedef struct mem_registration_ring { // TODO: Convert into a class and delete from pcx_mem.h all the Iop* functions and typdefs
  // TODO: Add documentation
  Iop usr_vec;  

  //PipeMem *usr_mem; TODO: Not used. Need to remove.
  PipeMem *tmpMem;
} mem_registration_ring_t;

// Performs data exchange between peers in ring.
// Sends data of size 'size' to 'peer' from 'send_buf' and 
// receives data of size 'size' from 'peer' to 'recv_buf'.
// 
// Args:
//    comm : Communicator that holds the rank ID of the current rank
//    peer : Rank ID of the rank that will take part in the data exchange
//           with the current rank
//    send_buf: The buffer that wil be sent to the rank with
//              rank id equals to 'peer'.
//    recv_buf: The buffer that recieve data from the rank with
//              rank id equals to 'peer'.
// 
int ring_exchange(void *comm, volatile void *send_buf, volatile void *recv_buf,
                 size_t size, uint32_t peer, uint32_t tag);

class StepCtx {
public:
  StepCtx() : outgoing_buf(NULL), umr_iov() {};
  ~StepCtx() {
    delete(this->outgoing_buf);
    freeIov(umr_iov);
  };
  Iov umr_iov;
  NetMem *outgoing_buf;
};

typedef struct rd_connections_ring {
  CommGraph *graph;
  LoopbackQp *lqp; // lqp stands for "Loopback Queue Pair"
  RingPair   *pqp; // pqp stands for "Pair Queue Pair"

  unsigned iters_cnt;
  StepCtx *iters;
} rd_connections_ring_t;

template <typename T> class PcxAllreduceRing : public Algorithm {
 public:

  //
  // Constructor
  // User uses an instance of this class when it wants to perform 
  // an All-Reduce operation on the elements in the ptrs vector.
  // The reduced result should be stored in ptrs[0].
  // In order to actually to perform the reduce the of elements, 
  // the user should call the PcxAllreduceRing::run() function.
  // 
  // Args:
  //    context : Struct that holds the communicator information. 
  //              The field 'rank' within the context is with the value 
  //              of the rank that uses the context.
  //    ptrs    : Vector of elements to reduce
  //    count   : The number of elements of type T to reduce
  //    fn      : The reduction function. Default is 'sum'.
  // 
  PcxAllreduceRing(
      const std::shared_ptr<Context> &context,
      const std::vector<T *> &ptrs, 
      const int count,
      const ReductionFunction<T> *fn = ReductionFunction<T>::sum)
      : Algorithm(context), 
        ptrs_(ptrs), 
        count_(count),
        bytes_(count_ * sizeof(T)),
        pieceSize_(bytes_ / contextSize_),
        fn_(fn) {

    PCX_RING_PRINT("Initializing PcxAllreduceRing");

    // In case the communicator is of size 1,
    // No need to reduce the ptrs vector, because
    // it's already reduced. The reduced result is
    // the first element in the vector (ptrs[0])
    if (this->contextSize_ == 1) {
      return;
    }

    // Step #1: 
    // Initialize verbs for all to use 
    PRINT("Starting PcxAllreduceRing");
    ibv_ = VerbCtx::getInstance();
    PCX_RING_PRINT("Verbs context initiated");

    // Step #2&3: 
    // Connect to the (recursive-doubling) 
    // iters and pre-post operations 
    connect_and_prepare();
    mone_ = 0;
  }

  // Destructor
  virtual ~PcxAllreduceRing() {
    PCX_RING_PRINT("Freeing UMR and freeing user memory");

    delete (rd_.lqp);
    delete (rd_.graph);
    delete (rd_.pqp);
    delete[](rd_.iters);

    // Deregister memory
    delete(mem_.tmpMem);
    PRINT("Freeing UMR and freeing user memory");
    freeIop(mem_.usr_vec);

    VerbCtx::remInstance();
  }

  void run() {
    debug_write_input();
    rd_.graph->mqp->qp->db();
    rd_.graph->mqp->qp->rearm();

    int res = 0;
    uint64_t count = 0;

    while (!res) {
      res = rd_.lqp->qp->poll();

      ++count;
      debug_hang_report(count);
    }
    debug_check_output();
    ++mone_;
  }

  void connect_and_prepare() { // TODO: Make this function private
    int vectors_to_reduce = ptrs_.size(); 

    unsigned step_count = contextSize_;
    unsigned comm_size = contextSize_;

    VerbCtx *ctx = (this->ibv_);

    // Create a single management QP
    rd_.graph = new CommGraph(ctx); // does lock
    CommGraph *sess = rd_.graph;
    PCX_RING_PRINT("Created management QP");

    // Step #2: Register existing memory buffers with UMR

    pipeline_ = RING_PIPELINE_DEPTH;
  
    // Find the maximal pipeline which devides the communicator
    // size without a reminder
    while (contextSize_ % pipeline_) {
      --pipeline_;
    }
    pipeline_ = contextSize_*2; // TODO: What is this? it overrides the loop that was before!!

    for (int i = 0; i < vectors_to_reduce; i++) {
      mem_.usr_vec.push_back(new PipeMem((void*)ptrs_[i], pieceSize_, 
                             (size_t)contextSize_, ibv_));
    }

    int temp_type = PCX_MEMORY_TYPE_MEMIC;
    temp_type = PCX_MEMORY_TYPE_HOST; // CHECK: Why is this patch needed? Why MEMIC cannot be used?

    mem_.tmpMem = new PipeMem(pieceSize_, pipeline_, ibv_, temp_type);

    // Create a loopback QP
    rd_.lqp = new LoopbackQp(sess);
    LoopbackQp *lqp = rd_.lqp;
    PCX_RING_PRINT("loopback connected");

    rd_.iters_cnt = contextSize_;
    rd_.iters = new StepCtx[contextSize_];
    if (!rd_.iters) {
      throw "malloc failed";
    }

    /* Establish a connection with each peer */
    uint32_t myRank = contextRank_;
    uint32_t slot1 = this->context_->nextSlot();
    uint32_t slot2 = this->context_->nextSlot();

    rd_.pqp = new RingPair(sess, &ring_exchange, (void *)&(this->context_), 
                           myRank, contextSize_ , slot1 , slot2 , mem_.tmpMem); 
    
    PCX_RING_PRINT("RC ring QPs created");

    RingQp* right = rd_.pqp->right;
    RingQp* left = rd_.pqp->left;

    for (unsigned step_idx = 0; step_idx < step_count; step_idx++) {
      size_t piece = (contextSize_ + myRank - step_idx) % contextSize_;
      for (int k = 0; k < vectors_to_reduce; ++k) {
        rd_.iters[step_idx].umr_iov.push_back(
          new RefMem((*mem_.usr_vec[k])[piece]));
      }
      if (step_idx > 0){
      	rd_.iters[step_idx].umr_iov.push_back(new RefMem(mem_.tmpMem->next()));
      }
      rd_.iters[step_idx].outgoing_buf = new UmrMem(rd_.iters[step_idx].umr_iov, 
                                                    ibv_);
    }
    PCX_RING_PRINT("UMR registration done");

    PCX_RING_PRINT("Starting All-Reduce");
    PCX_RING_PRINT("Starting Scatter-Reduce stage");

    int credits = pipeline_;

    if (credits>1){
      right->reduce_write(rd_.iters[0].outgoing_buf, 0, vectors_to_reduce, MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
      --credits;
      right->reduce_write_cmpl(rd_.iters[0].outgoing_buf, 0, vectors_to_reduce, MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
      sess->wait_send(right);
      left->sendCredit();
      sess->wait(right);

      // Initialize number of credits
      credits = pipeline_;
    }
    sess->wait(left);

    PCX_RING_PRINT("Performing first reduce in the Reduce-Scatter stage");   
 
    for (unsigned step_idx = 1; step_idx < step_count; step_idx++) {
      if (credits==1){
        right->reduce_write_cmpl(rd_.iters[step_idx].outgoing_buf, step_idx, (vectors_to_reduce+1) , MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32); 
        sess->wait_send(right);
        left->sendCredit();
        sess->wait(right);
        credits = pipeline_;
      } else {
        right->reduce_write(rd_.iters[step_idx].outgoing_buf, step_idx, (vectors_to_reduce+1) , MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
        --credits;
      }
      sess->wait(left);
    }

    PCX_RING_PRINT("Reduce-Scatter stage done");


    // Done with the AllReduce-Scatter
    size_t last_frag = (step_count-1);

    for (unsigned step_idx = 0; step_idx < step_count; step_idx++) {
      RefMem newVal((*mem_.tmpMem)[last_frag]);

      size_t piece = (step_idx + myRank) % step_count;
      if (credits==1){
        right->writeCmpl(&newVal, step_count + step_idx );
        for (uint32_t buf_idx = 0; buf_idx < vectors_to_reduce; buf_idx++) {
          lqp->write(&newVal, rd_.iters[step_idx].umr_iov[buf_idx]);
        }
        sess->wait_send(right);
        sess->wait(lqp);
        left->sendCredit();
        sess->wait(right); //for credit
        credits = pipeline_;
      } else {
        right->write(&newVal, step_count + step_idx);
        for (uint32_t buf_idx = 0; buf_idx < vectors_to_reduce; buf_idx++) {
          lqp->write(&newVal, rd_.iters[step_idx].umr_iov[buf_idx]);
        }
      }
      sess->wait(left); //for data
      ++last_frag;
    }

    PCX_RING_PRINT("All-Gather stage done");

    if (credits != pipeline_){
      left->sendCredit();
      sess->wait(right);
      PCX_RING_PRINT("Returned all credits to peer");
    }

    PCX_RING_PRINT("Graph building stage done");

    PCX_RING_PRINT("connect_and_prepare DONE");
  }

  // Debug function // TODO: Make this function private!
  void debug_write_input() { 
  #ifdef VALIDITY_CHECK
    for (int i = 0; i < ptrs_.size(); ++i) {
      // fprintf(stderr, "Input %d:\n",i);
      float *buf = (float *)ptrs_[i];
      for (int k = 0; k < count_; ++k) {
        buf[k] = ((float)k + i) + contextRank_ + mone_;
      }
      // print_values(buf, count_);
    }
  #endif // VALIDITY_CHECK
  }

  // Debug function // TODO: Make this function private!
  void debug_hang_report(uint64_t &count) {
  #ifdef HANG_REPORT
    if (count == 1000000000) {
      fprintf(stderr, "iteration: %d\n", mone_);
      fprintf(stderr, "poll cnt: %d\n", rd_.lqp->qp->get_poll_cnt());
      fprintf(stderr, "managment qp:\n");
      rd_.graph->mqp->print();
      fprintf(stderr, "loopback qp:\n");
      rd_.lqp->print();
      fprintf(stderr, "right qp:\n");
      rd_.pqp->right->print();
      fprintf(stderr, "left qp:\n");
      rd_.pqp->left->print();
    }
  #endif // HANG_REPORT
  }

  // Debug function // TODO: Make this function private!
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
            ((k + mone_) * 2 + ptrs_.size() - 1) * ptrs_.size() / 2;
        int expected_max =
            ((k + mone_ + contextSize_ - 1) * 2 + ptrs_.size() - 1) *
            ptrs_.size() / 2;
        float expected_result =
            (float)(expected_base + expected_max) * contextSize_ / 2;
        float result = buf[k];
        if (result != expected_result) {
          fprintf(stderr,
                  "ERROR: In Iteration %d\n expected: %.2f, got: %.2f\n", mone_,
                  expected_result, result);
          for (int i = 0; i < ptrs_.size(); ++i) {
            fprintf(stderr, "Input %d:\n", i);
            float buf[count_];
            for (int k = 0; k < count_; ++k) {
              buf[k] = ((float)k + i) + contextRank_ + mone_;
            }
            print_values(buf, count_);
          }
          mem_.tmpMem->print();
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
  #endif // VALIDITY_CHECK
  }


 protected:
  // Vector of elements to reduce.
  // Initialized in the constructor.
  std::vector<T*> ptrs_;
 
  // Number of elements that will be reduced.
  // Each element is of type T
  // Initialized in the constructor.
  const int count_;

  // Total amount of bytes of all elements
  // Initialized in the constructor.
  const int bytes_;

  // The reduction function to use when performing the reduce 
  // Initialized in the constructor.
  const ReductionFunction<T> *fn_;

  VerbCtx *ibv_;
  mem_registration_ring_t mem_;
  rd_connections_ring_t rd_;

  int mone_;

  // TODO: Add documentation. What is this?
  int pipeline_ = RING_PIPELINE_DEPTH; // TODO: Consider converting into 'static constexpr int'

  // The size of each chunk that will be moved through
  // ring throughout the run of the algorithm
  size_t pieceSize_;
};

} // namespace gloo
