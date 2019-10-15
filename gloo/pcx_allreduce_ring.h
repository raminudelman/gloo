/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found inpqp* LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "third-party/pcx/pcx_allreduce_alg_chunked_ring.h"

#include "gloo/algorithm.h"
#include "gloo/context.h"

#include <alloca.h>
#include <stddef.h>
#include <string.h>
#include <ctime>
#include <vector>

namespace gloo {

template <typename T>
class PcxAllreduceRing : public Algorithm {
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
        pcxAlg(contextSize_, 
               contextRank_, 
               ptrs, 
               count, 
               this->context_->nextSlot(), 
               this->context_->nextSlot(), 
               (void *)&(this->context_)) {}

  void run() {
    pcxAlg.run();
  }


private:
  PcxAllreduceChunkedRing<T> pcxAlg;
};

} // namespace gloo
