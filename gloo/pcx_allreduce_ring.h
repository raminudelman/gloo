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
#include "gloo/pcx_allreduce_common.h"

#include <ctime>
#include <vector>

#define DEBUG
#ifdef DEBUG
#define PCX_RING_PRINT(args...) fprintf(stderr, "(%s: %d) in function %s: " \
                       ,__FILE__,__LINE__,__func__); fprintf(stderr, args)
#else
#define PCX_RING_PRINT(args...)
#endif

namespace gloo {

typedef struct mem_registration_ring { // TODO: Convert into a class and delete from pcx_mem.h all the Iop* functions and typdefs
  // TODO: Add documentation
  Iop usr_vec;  

  PipeMem *tmpMem;
} mem_registration_ring_t;

// Performs data exchange between peers in ring.
// Sends data of size 'size' to 'peer' from 'send_buf' and 
// receives data of size 'size' from 'peer' to 'recv_buf'.
// This function is used for connecting the QPs of two ranks.
// After the execution of this function, the ranks will be able to communicate
// via the two QPs (each rank through it's own QP on it's own end)
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
  Iov umr_iov; // Iov == Input/Output Vector, UMR is because the user's buffer is not contigious and we convert it to a UMR.
  NetMem *outgoing_buf;  // The buffer which contains the result of the reduce and which will be sent to the peer rank
};

typedef struct rd_connections_ring {
  CommGraph *graph;
  LoopbackQp *lqp; // lqp stands for "Loopback Queue Pair"
  RingPair   *pqp; // pqp stands for "Pair Queue Pair"

  // Holds the number of iterations that will be executed during the All-Reduce
  // algorithm
  unsigned iters_cnt;

  // Each element in the array holds all the data structure that the algorithm
  // operates on during each step of the algorithm.
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

    PCX_RING_PRINT("Initializing PcxAllreduceRing \n");

    // In case the communicator is of size 1,
    // No need to reduce the ptrs vector, because
    // it's already reduced. The reduced result is
    // the first element in the ptrs vector (ptrs[0])
    if (this->contextSize_ == 1) {
      return;
    }

    // Step #1: 
    // Initialize verbs context (choose IB device, open PD, etc.)
    ibv_ctx_ = VerbCtx::getInstance();
    PCX_RING_PRINT("Verbs context initiated \n");

    // Step #2 & #3:  // TODO: Improve the comment/documentation
    // Connect to the (recursive-doubling) 
    // iters and pre-post operations 
    connect_and_prepare();
    mone_ = 0;
  }

  // Destructor
  virtual ~PcxAllreduceRing() {
    PCX_RING_PRINT("Freeing UMR and freeing user memory \n");

    delete (rd_.lqp);
    delete (rd_.graph);
    delete (rd_.pqp);
    delete[](rd_.iters);

    // Deregister memory
    delete(mem_.tmpMem);
    freeIop(mem_.usr_vec);

    VerbCtx::remInstance();
  }

  void run() {
    debug_write_input();
    rd_.graph->mqp->qp->db();

    // Calling the rearm after the collective operation started (using the 
    // DoorBell) makes the rearm process to run in parallel with the 
    // collective algorithm.
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
    // The number of vectors to reduce.
    // Each vector has count_ elements.
    int vectors_to_reduce = ptrs_.size(); 

    // step_count holds the number of iterations that the ring algorithm should
    // perform in reduce-scatter stage, and in all-gather stage.
    // During reduce-scatter it will perform step_count iterations,
    // and in all-gather stage it will perform additional step_count iterations.
    unsigned step_count = contextSize_;

    // Create a single management QP
    rd_.graph = new CommGraph(ibv_ctx_); // locks the mutex in the ctx
    CommGraph *sess = rd_.graph;
    PCX_RING_PRINT("Created management QP \n");

    // Step #2: Register existing memory buffers with UMR

    // Register (ibv_reg_mr) the users data buffers.
    for (int i = 0; i < vectors_to_reduce; i++) {
      mem_.usr_vec.push_back(new PipeMem((void*)ptrs_[i], pieceSize_, 
                             (size_t)contextSize_, ibv_ctx_));
    }

    int temp_type = PCX_MEMORY_TYPE_MEMIC;
    temp_type = PCX_MEMORY_TYPE_HOST; // CHECK: Why is this patch needed? Why MEMIC cannot be used? MEMIC worked but during debug this code was added to prevent other bugs or compilcations. MEMIC is not that important and did not help performance too much because after some point the MEMIC region is not enought and it wil fall back to host mem anyway.

    pipeline_ = RING_PIPELINE_DEPTH;
  
    // Find the maximal pipeline which devides the communicator
    // size without a reminder
    while (contextSize_ % pipeline_) {
      --pipeline_;
    }
    pipeline_ = contextSize_*2; // TODO: What is this? it overrides the loop that was before!!

    // The tmpMem will be used for "incoming" messages from the qps, this buffer is of size pipeline and each element in the buffer is with size peicesize.
    mem_.tmpMem = new PipeMem(pieceSize_, pipeline_, ibv_ctx_, temp_type);

    // Create a loopback QP - used for DMA inside the container itself. 
    // Instead of using MemCpy and CudaMemCopy, the memory is copied via the NIC.
    rd_.lqp = new LoopbackQp(sess);
    LoopbackQp *lqp = rd_.lqp;
    PCX_RING_PRINT("Loopback created \n");

    // Establish a connection with each peer
    uint32_t myRank = contextRank_;
    uint32_t slot1 = this->context_->nextSlot();
    uint32_t slot2 = this->context_->nextSlot();

    rd_.pqp = new RingPair(sess, &ring_exchange, (void *)&(this->context_), 
                           myRank, contextSize_ , slot1 , slot2 , mem_.tmpMem); 
    PCX_RING_PRINT("RC ring QPs created \n");

    // Allocating a data structure for every step in the algorithm.
    // The data structure will hold all the required data buffers for the
    // step in the algorithm
    rd_.iters_cnt = contextSize_;
    rd_.iters = new StepCtx[contextSize_];
    if (!rd_.iters) {
      throw "malloc failed";
    }

    // For every step in the ring algorithm, we create a single umr vector
    // that combines all the corresponding pieces from the usr_vec.
    for (unsigned step_idx = 0; step_idx < step_count; step_idx++) {
      size_t piece = (contextSize_ + myRank - step_idx) % contextSize_;
      for (int k = 0; k < vectors_to_reduce; ++k) {
        rd_.iters[step_idx].umr_iov.push_back(
          new RefMem((*mem_.usr_vec[k])[piece]));
      }
      if (step_idx > 0){
      	rd_.iters[step_idx].umr_iov.push_back(new RefMem(mem_.tmpMem->next())); // The next() operation is cyclic.
      }
      rd_.iters[step_idx].outgoing_buf = new UmrMem(rd_.iters[step_idx].umr_iov, 
                                                    ibv_ctx_);
    }
    PCX_RING_PRINT("UMR registration done \n");

    // For convenience, we will define local variables
    RingQp* right = rd_.pqp->right;
    RingQp* left = rd_.pqp->left;

    PCX_RING_PRINT("Starting All-Reduce \n");
    PCX_RING_PRINT("Starting Scatter-Reduce stage \n");

    int credits = pipeline_;

    if (credits>1){
      // reduce_write means that we perform reduce and perform RDMA write to the
      // next rank. The RDMA write will send the outgoing_buf to the incoming 
      // buffer (wihch is the tmpMem) of the destination rank.
      right->reduce_write(rd_.iters[0].outgoing_buf, 0, vectors_to_reduce, MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
      --credits;
    } else { // Credits == 1
      // reduce_write_cmpl means that we perform reduce and perform RDMA write to the next rank and require a completion for the RDMA write.
      right->reduce_write_cmpl(rd_.iters[0].outgoing_buf, 0, vectors_to_reduce, MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
      sess->wait_send(right);
      left->send_credit();
      sess->wait(right);

      // Initialize number of credits
      credits = pipeline_;
    }
    // Once a rank sent a message from the sending QP (to the rank to the right), 
    // the rank knows it should wait for the message from the left QP  the 
    sess->wait(left);

    PCX_RING_PRINT("Performed first reduce in the Reduce-Scatter stage \n");   
 
    // The first reduce (first step in the ring algorithm)
    for (unsigned step_idx = 1; step_idx < step_count; step_idx++) {
      if (credits==1){
        right->reduce_write_cmpl(rd_.iters[step_idx].outgoing_buf, step_idx, (vectors_to_reduce+1) , MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32); 
        sess->wait_send(right);
        left->send_credit(); // Notifying the left rank that it can continue sending new data
        sess->wait(right); // Waiting for the rank from the right to realse a credit that mean that our rank can continue sending the data to the right.
        credits = pipeline_;
      } else {
        right->reduce_write(rd_.iters[step_idx].outgoing_buf, step_idx, (vectors_to_reduce+1) , MLX5DV_VECTOR_CALC_OP_ADD, MLX5DV_VECTOR_CALC_DATA_TYPE_FLOAT32);
        --credits;
      }
      sess->wait(left);
    }

    PCX_RING_PRINT("Reduce-Scatter stage done \n");

    // Done with the AllReduce-Scatter. Every rank has a peice of the final
    // result stored in it's tmpMem (in one of the slots).

    // Start the All-Gather step!!
    size_t last_frag = (step_count-1);

    for (unsigned step_idx = 0; step_idx < step_count; step_idx++) {
      // Ref mem does not alocate new memory...
      RefMem newVal((*mem_.tmpMem)[last_frag]); // Cyclic pipe...there is wraparound

      size_t piece = (step_idx + myRank) % step_count;
      if (credits==1){
        right->write_cmpl(&newVal, step_count + step_idx );
        // Copying "the reduce result from ptrs[0] to all ptrs[i]"
        for (uint32_t buf_idx = 0; buf_idx < vectors_to_reduce; buf_idx++) {
          lqp->write(&newVal, rd_.iters[step_idx].umr_iov[buf_idx]);
        }
        sess->wait_send(right);
        sess->wait(lqp); //Waiting for the receive to finish in the loopback QP
        left->send_credit();
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

    PCX_RING_PRINT("All-Gather stage done \n");

    // Making the NIC wait for the last credit although the user already got the reduce result from the lpq because in the run, we poll only the lpq.
    if (credits != pipeline_){
      left->send_credit();
      sess->wait(right);
      PCX_RING_PRINT("Returned all credits to peer \n");
    }

    rd_.graph->finish(); // unlocks the mutex in the ctx
    PCX_RING_PRINT("Graph building stage done \n");

    PCX_RING_PRINT("connect_and_prepare DONE \n");
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
  // Vector of required elements to reduce. 
  // Assume ptrs_ vector is of size N.
  // Each element in the ptrs_ vector is of type T* (pointer to an array).
  // Every T* element is a pointer to a vector/array with count_ elements.
  // and every element is of type T.
  // The ptrs_ vector is initialized in the constructor.
  //
  // The ptrs_ vector can be visualized as follows:
  //
  //    ptrs[0] , ptrs[1] , ptrs[2] , ... , ptrs[N-1]
  //
  // The ptrs_[i] can be visualized as follows (assume I = count_):
  //
  //    ptrs[i]:   Ti[0] , Ti[1] , Ti[2] , ... , Ti[count_-1]
  //
  // Where every Ti[j] element is an element of type T.
  // Finally, ptrs_ vector can be visualized as follows:
  //
  //   {     ptrs_[0]      },{         ptrs_[1]  },...,{      ptrs_[N-1]       }
  //    [T0[0],...,T0[I-1]] , [T1[0],...,T1[I-1]] ,..., [TN_1[0],...,TN_1[I-1]]
  //
  // The ptrs_ vector can be seen as a matrix with dimentions of ptrs_.size()
  // raws and count_ columns. Each cell [i][j] in the matrix is of type T.
  // Each raw in the matrix contains a single element in the ptrs_ vector and 
  // every column represents the element of type T
  //
  // The reduce result can be presented easly via the matrix view.
  // The reduce operation is performed on the column, meaning for every column j
  // all the raws 0 to N-1 are reduced and the same result stored in all the 
  // raws of column j.
  std::vector<T*> ptrs_;
 
  // Number of elements in each element of the ptrs_ vector.
  // Notice that every element in the ptrs_ vector may be also a vector/array)
  // that will be reduced.
  // The count_ variable is initialized in the constructor.
  const int count_;

  // Total amount of bytes of all elements in a single element in ptrs_ vector.
  // Initialized in the constructor.
  // For example if a single ptrs_ element is a vector of 5 elements each of
  // size T, then bytes_ will be equal to 5*size_in_bytes(T).
  const int bytes_;

  // The reduction function to use when performing the reduce operation.
  // Initialized in the constructor.
  const ReductionFunction<T> *fn_;

  VerbCtx *ibv_ctx_;

  mem_registration_ring_t mem_;
  rd_connections_ring_t rd_;

  // Counts how many times the algorithm ran (for debug reasons). 
  int mone_;

  // Holds the number of data pieces (each peace is of size pieceSize_) 
  // that a peer rank (in the ring) can send to this rank without waiting for 
  // the rank to notify that the buffers used to recieve the data can be reused.
  int pipeline_ = RING_PIPELINE_DEPTH; // TODO: Consider converting into 'static constexpr int'

  // The size of each chunk that will be moved through
  // ring throughout the run of the algorithm
  size_t pieceSize_;
};

} // namespace gloo
