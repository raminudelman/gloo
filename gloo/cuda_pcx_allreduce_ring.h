/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found inpqp* LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/cuda.h"
#include "gloo/cuda_workspace.h"
#include "pcx_allreduce_ring.h"

namespace gloo
{

template <typename T>
class CudaPcxAllreduceRing : public PcxAllreduceRing<T>
{
public:
    CudaPcxAllreduceRing(
        const std::shared_ptr<Context> &context,
        const std::vector<T *> &ptrs,
        const int count,
        const std::vector<cudaStream_t> &streams)
        : PcxAllreduceRing<T>(context,
                              ptrs,
                              count,
                              ReductionFunction<T>::sum),
          fn_cuda_(CudaReductionFunction<T>::sum)
    {
        auto newStream = true;
        if (streams.size() > 0)
        {
            GLOO_ENFORCE_EQ(streams.size(), ptrs.size());
            newStream = false;
        }

        for (auto i = 0; i < ptrs.size(); i++)
        {
            auto ptr = CudaDevicePointer<T>::create(ptrs[i], count);
            if (newStream)
            {
                streams_.push_back(CudaStream(ptr.getDeviceID()));
            }
            else
            {
                streams_.push_back(CudaStream(ptr.getDeviceID(), streams[i]));
            }
            //devicePtrs_.push_back(std::move(ptr));
        }
    }
    void run()
    {
        for (auto i = 0; i < streams_.size(); i++)
        {
            cudaStreamSynchronize(streams_[i].getStream());
        }
        PcxAllreduceRing<T>::run();
    }

protected:
    std::vector<CudaStream> streams_;
    const CudaReductionFunction<T> *fn_cuda_;
};
} // namespace gloo