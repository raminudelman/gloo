/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <memory>
#include <vector>

#include "gloo/cuda_allreduce_bcube.h"
#include "gloo/cuda_allreduce_halving_doubling.h"
#include "gloo/cuda_allreduce_halving_doubling_pipelined.h"
#include "gloo/pcx_allreduce_king.h"
#include "gloo/cuda_allreduce_ring.h"
#include "gloo/cuda_pcx_allreduce_ring.h"
#include "gloo/cuda_pcx_allreduce_king.h"
#include "gloo/cuda_allreduce_ring_chunked.h"
#include "gloo/test/cuda_base_test.h"

namespace gloo {
namespace test {
namespace {

// Function to instantiate and run algorithm.
// The function gets as arguments: context, ptrs, count and streams
// and the returns an Algorithm.
using Func = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams);

using Func16 = std::unique_ptr<::gloo::Algorithm>(
    std::shared_ptr<::gloo::Context>&,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>, int>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

// Test case
class CudaAllreduceTest : public CudaBaseTest,
                          public ::testing::WithParamInterface<Param> {
 public:
  void assertResult(CudaFixture<float>& fixture) {
    fixture.copyToHost();
    fixture.checkAllreduceResult();
  }
};

class CudaAllreduceTestHP : public CudaBaseTest,
                            public ::testing::WithParamInterface<ParamHP> {
 public:
  void assertResult(CudaFixture<float16>& fixture) {
    fixture.copyToHost();
    fixture.checkAllreduceResult();
  }
};

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRing<float>(context, ptrs, count, streams));
};

static std::function<Func> allreducePcxRing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaPcxAllreduceRing<float>(context, ptrs, count, streams));
};

static std::function<Func16> allreduceRingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRing<float16>(context, ptrs, count, streams));
};

static std::function<Func16> allreducePcxRingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::PcxAllreduceRing<float16>(context, ptrs, count));
};

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
    new ::gloo::CudaAllreduceRingChunked<float>(context, ptrs, count, streams));
};

static std::function<Func16> allreduceRingChunkedHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceRingChunked<float16>(
          context, ptrs, count, streams));
};

static std::function<Func> allreduceHalvingDoubling = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoubling<float>(
          context, ptrs, count, streams));
};

static std::function<Func> allreducePcxKing = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaPcxAllreduceKing<float>(context, ptrs, count, streams));
};

static std::function<Func> allreduceBcube =
    [](std::shared_ptr<::gloo::Context>& context,
       std::vector<float*> ptrs,
       int count,
       std::vector<cudaStream_t> streams) {
      return std::unique_ptr<::gloo::Algorithm>(
          new ::gloo::CudaAllreduceBcube<float>(context, ptrs, count, streams));
    };

static std::function<Func16> allreduceHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoubling<float16>(
          context, ptrs, count, streams));
};

static std::function<Func16> allreducePcxKingHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::PcxAllreduceKing<float16>(context, ptrs, count));
};

static std::function<Func> allreduceHalvingDoublingPipelined = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoublingPipelined<float>(
          context, ptrs, count, streams));
};

static std::function<Func16> allreduceHalvingDoublingPipelinedHP = [](
    std::shared_ptr<::gloo::Context>& context,
    std::vector<float16*> ptrs,
    int count,
    std::vector<cudaStream_t> streams) {
  return std::unique_ptr<::gloo::Algorithm>(
      new ::gloo::CudaAllreduceHalvingDoublingPipelined<float16>(
          context, ptrs, count, streams));
};

// SinglePointer test, tests the AllReduce algorithm to reduce
// a single element in the ptrs vector. Notice that the single
// element in the ptrs vector can be by itself a vector of several
// elements. The amount of elements that consist a single element
// of the ptrs vector is determined with 'count'.
TEST_P(CudaAllreduceTest, SinglePointer) {
  // Context Size (Number of ranks)
  auto size = std::get<0>(GetParam());

  // Number of elements within a single ptrs element.  
  auto count = std::get<1>(GetParam());

  // Algorithm to use for AllReduce operation
  auto fn = std::get<2>(GetParam());

  // Base used as a parameter for BCube algorithm
  // and stored in the context object
  auto base = std::get<3>(GetParam());

  fprintf(stderr, "SinglePointerTest: Context Size = %d. Count = %d \n", size, count);

  spawn(
      size,
      [&](std::shared_ptr<Context> context) {
        // Run algorithm
        auto fixture = CudaFixture<float>(context, 1, count);
        auto ptrs = fixture.getCudaPointers();
        auto algorithm = fn(context, ptrs, count, {});
        fixture.assignValues();
        algorithm->run();
        // Verify result
        assertResult(fixture);
      },
      base);
}

// In MultiPointer test, the ptrs vector (that is defined later)
// contains several elements versus only a single element
// in the SinglePointer test.
TEST_P(CudaAllreduceTest, MultiPointer) {
  // Context Size (Number of ranks)
  auto size = std::get<0>(GetParam());

  // Number of elements within a single ptrs element.  
  auto count = std::get<1>(GetParam());

  // Algorithm to use for AllReduce operation
  auto fn = std::get<2>(GetParam());

  // Base used as a parameter for BCube algorithm
  // and stored in the context object
  auto base = std::get<3>(GetParam());

  fprintf(stderr, "MultiPointerTest: Context Size = %d. Count = %d \n", size, count);

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, cudaNumDevices(), count);
      auto ptrs = fixture.getCudaPointers();
      auto algorithm = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm->run();

      // Verify result
      assertResult(fixture);
    }, base);
}

TEST_P(CudaAllreduceTest, MultiPointerAsync) {
  auto size = std::get<0>(GetParam());
  auto count = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());
  auto base = std::get<3>(GetParam());

  spawn(size, [&](std::shared_ptr<Context> context) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, cudaNumDevices(), count);
      auto ptrs = fixture.getCudaPointers();
      auto streams = fixture.getCudaStreams();
      auto algorithm = fn(context, ptrs, count, streams);
      fixture.assignValuesAsync();
      algorithm->run();

      // Verify result
      fixture.synchronizeCudaStreams();
      assertResult(fixture);
    }, base);
}

TEST_F(CudaAllreduceTest, MultipleAlgorithms) {
  auto size = 4;
  auto count = 1000;
  auto fns = {allreduceRing,
             allreducePcxRing,
             allreducePcxKing,
             allreduceRingChunked,
             allreduceHalvingDoubling,
             allreduceHalvingDoublingPipelined};

  spawn(size, [&](std::shared_ptr<Context> context) {
    for (const auto& fn : fns) {
      // Run algorithm
      auto fixture = CudaFixture<float>(context, 1, count);
      auto ptrs = fixture.getCudaPointers();

      auto algorithm = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm->run();

      // Verify result
      assertResult(fixture);

      auto algorithm2 = fn(context, ptrs, count, {});
      fixture.assignValues();
      algorithm2->run();

      // Verify result
      assertResult(fixture);
    }
  });
}

TEST_F(CudaAllreduceTestHP, MultipleAlgorithmsHP) {
  auto size = 4;
  auto count = 128;
  auto fns = {allreduceRingHP,
             //allreducePcxRingHP, // TODO: PCX Does not support half precision because Vector-CALC hardware does not support float16. Need to un-commnet when hardware supports.
             //allreducePcxKingHP, // TODO: PCX Does not support half precision because Vector-CALC hardware does not support float16. Need to un-commnet when hardware supports.
             allreduceRingChunkedHP,
             allreduceHalvingDoublingHP,
             allreduceHalvingDoublingPipelinedHP};
  spawn(size, [&](std::shared_ptr<Context> context) {
      for (const auto& fn : fns) {
        // Run algorithm
        auto fixture = CudaFixture<float16>(context, 1, count);
        auto ptrs = fixture.getCudaPointers();

        auto algorithm = fn(context, ptrs, count, {});
        fixture.assignValues();
        algorithm->run();

        // Verify result
        assertResult(fixture);
      }
    });
}

std::vector<int> genMemorySizes() {
  std::vector<int> v;
  v.push_back(sizeof(float));
  v.push_back(100);
  v.push_back(1000);
  v.push_back(10000);
  return v;
}

INSTANTIATE_TEST_CASE_P(
    AllreduceRing,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(1, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRing),
      ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreducePcxRing,
    CudaAllreduceTest,
    ::testing::Combine(
      // TODO: Make Ring work with all possible sizes of context (especially for a context of size == 1).
      ::testing::ValuesIn(std::vector<int>({2, 4})), //::testing::Range(2, 16,1), // Start, End, Step size
                                                     // Sizes that does not works:
                                                     //   size = 1,        fails on:
                                                     //   size = 3,        fails on:
                                                     //   size= {1,5-inf}, fails on: Get simply stuck with no output after the getInstance() prints in verbs_ctx.cc
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreducePcxRing),
      ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceRingChunked,
    CudaAllreduceTest,
    ::testing::Combine(
      ::testing::Range(1, 16),
      ::testing::ValuesIn(genMemorySizes()),
      ::testing::Values(allreduceRingChunked),
      ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoubling,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoubling),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreducePcxKing,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          // TODO: Make Ring work with all possible sizes of context (especially for a context of size == 1).
          std::vector<int>({2, 4, 8, 16, 32})), // TODO: King currently support only context size which is power of 2 (meaning 1,2,4,8, etc.).
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreducePcxKing),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase2,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 4, 8, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(2)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase3,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>({1, 3, 9, 27})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(3)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase4,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>({1, 4, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(4)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoublingPipelined,
    CudaAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoublingPipelined),
        ::testing::Values(0)));

} // namespace
} // namespace test
} // namespace gloo
