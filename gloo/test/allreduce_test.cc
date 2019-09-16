/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>

#include <functional>
#include <thread>
#include <vector>

#include "gloo/allreduce.h"
#include "gloo/allreduce_bcube.h"
#include "gloo/allreduce_halving_doubling.h"
#include "gloo/pcx_allreduce_king.h"
#include "gloo/allreduce_ring.h"
#include "gloo/pcx_allreduce_ring.h"
#include "gloo/allreduce_ring_chunked.h"
#include "gloo/test/base_test.h"

namespace gloo {
namespace test {
namespace {

// RAII handle for aligned buffer
template <typename T>
std::vector<T, aligned_allocator<T, kBufferAlignment>> newBuffer(int size) {
  return std::vector<T, aligned_allocator<T, kBufferAlignment>>(size);
}

// Function to instantiate and run algorithm.
using Func = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat);

using Func16 = void(
    std::shared_ptr<::gloo::Context>,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat);

// Test parameterization.
using Param = std::tuple<int, int, std::function<Func>, int>;
using ParamHP = std::tuple<int, int, std::function<Func16>>;

template <typename Algorithm>
class AllreduceConstructorTest : public BaseTest {
};

typedef ::testing::Types<
  AllreduceRing<float>,
  PcxAllreduceRing<float>,
  AllreduceRingChunked<float> > AllreduceTypes;
TYPED_TEST_CASE(AllreduceConstructorTest, AllreduceTypes);

TYPED_TEST(AllreduceConstructorTest, InlinePointers) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      TypeParam algorithm(
        context,
        {&f},
        1);
    });
}

TYPED_TEST(AllreduceConstructorTest, SpecifyReductionFunction) {
  this->spawn(2, [&](std::shared_ptr<Context> context) {
      float f = 1.0f;
      std::vector<float*> ptrs = {&f};
      TypeParam algorithm(
        context,
        ptrs,
        ptrs.size(),
        ReductionFunction<float>::product);
    });
}

static std::function<Func> allreduceRing = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceRing<float> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func> allreducePcxRing = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::PcxAllreduceRing<float> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func16> allreduceRingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceRing<float16> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func16> allreducePcxRingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::PcxAllreduceRing<float16> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func> allreduceRingChunked = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceRingChunked<float> algorithm(
      context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func16> allreduceRingChunkedHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceRingChunked<float16> algorithm(
      context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func> allreduceHalvingDoubling = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceHalvingDoubling<float> algorithm(
      context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func> allreducePcxKing = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::PcxAllreduceKing<float> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func> allreduceBcube = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceBcube<float> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func16> allreduceHalvingDoublingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::AllreduceHalvingDoubling<float16> algorithm(
      context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

static std::function<Func16> allreducePcxKingHP = [](
    std::shared_ptr<::gloo::Context> context,
    std::vector<float16*> dataPtrs,
    int dataSize,
    int repeat = 1) {
  ::gloo::PcxAllreduceKing<float16> algorithm(context, dataPtrs, dataSize);
  for (int i = 0; i < repeat; ++i) {
    algorithm.run();
  }
};

// Test fixture.
class AllreduceTest : public BaseTest,
                      public ::testing::WithParamInterface<Param> {};

class AllreduceTestHP : public BaseTest,
                        public ::testing::WithParamInterface<ParamHP> {};

TEST_P(AllreduceTest, SinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());
  auto base = std::get<3>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }

    fn(context, std::vector<float*>{ptr}, dataSize, 1);

    auto expected = (contextSize * (contextSize - 1)) / 2;
    for (int i = 0; i < dataSize; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
    }
  }, base);
}

TEST_F(AllreduceTest, MultipleAlgorithms) {
  // contextSize defines the number of ranks that will
  // participate in the reduction operation
  auto contextSize = 4;

  // dataSize defines the number of elements in every item in the ptr vector.
  // In this test, the ptr is a vector with a single element.
  auto dataSize = 1000;
  
  // fns determines which algorithms will be checked.
  // Each AllReduce operation will use a single algorithm. 
  auto fns = {
              allreduceRing,
              allreducePcxRing,
              allreduceRingChunked,
              allreduceHalvingDoubling,
              allreducePcxKing,
              allreduceBcube
              };

  // Spawn threads. Each thread is a different rank.
  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();

    // Performing AllReduce using every algorithm that stored in fns.
    for (const auto& fn : fns) {
      // Each AllReduce will reduce a vector with size equals to dataSize.
      // The vector of rank with rank ID 'rank_id' will have all elements
      // equal to 'rank_id', meaning the vectors of N ranks will be as follows:
      //     rank_0: <0,0,0,...,0> // Total of dataSize elements
      //     rank_1: <1,1,1,...,1> // Total of dataSize elements
      //     rank_2: <2,2,2,...,2> // Total of dataSize elements
      //     ...
      //     rank_N-1: <N-1,N-1,N-1,...,N-1> // Total of dataSize elements
      //
      // When using sum as a reduction function, after the AllReduce operation,
      // each rank should have a vector that looks as follows:
      //
      //     rank_i: <0+1+2+...+N,0+1+2+...+N,...,0+1+2+...+N>
      //
      // Or by using the sum of am arithmetic sequence: S_n = (N*(N-1))/2
      //  
      //     rank_i: <(N*(N-1))/2,(N*(N-1))/2,...,(N*(N-1))/2>
      //
      //
      // For contextSize=4, N will be equal to 4
      // so the sum will be equal to 4*3/2 = 6
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize, 1);

      auto expected = (contextSize * (contextSize - 1)) / 2;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i << ". Rank: " << contextRank;
      }

      // Now we will populate the ptr vector with the same data but we will
      // run the allreduce algorithm twice in a row. After the first run,
      // all the elements if ptr will be equal to the sum of all ranks, and
      // the second run will perform another allreduce that will cause each
      // element in the ptr vector to be equal the sum of all ranks and then
      // multiplied by the amount of ranks.
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float*>{ptr}, dataSize, 2);

      expected = ((contextSize * (contextSize - 1)) / 2) * contextSize;
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i << ". Rank: " << contextRank;
      }
    }
  });
}

TEST_F(AllreduceTestHP, MultipleAlgorithmsHP) {
  int contextSize = 4;
  auto dataSize = 1024;
  auto fns = {allreduceRingHP,
              //allreducePcxRingHP, // TODO: PCX Does not support half precision because Vector-CALC hardware does not support float16. Need to un-commnet when hardware supports.
              //allreducePcxKingHP, // TODO: PCX Does not support half precision because Vector-CALC hardware does not support float16. Need to un-commnet when hardware supports.
              allreduceRingChunkedHP,
              allreduceHalvingDoublingHP};

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float16>(dataSize);
    auto* ptr = buffer.data();
    for (const auto& fn : fns) {
      for (int i = 0; i < dataSize; i++) {
        ptr[i] = contextRank;
      }

      fn(context, std::vector<float16*>{ptr}, dataSize, 1);

      float16 expected(contextSize * (contextSize - 1) / 2);
      for (int i = 0; i < dataSize; i++) {
        ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
      }
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
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(1, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRing),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreducePcxRing,
    AllreduceTest,
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
    AllreduceTest,
    ::testing::Combine(
        ::testing::Range(1, 16),
        ::testing::ValuesIn(genMemorySizes()),
        ::testing::Values(allreduceRingChunked),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreduceHalvingDoubling,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 24, 32})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceHalvingDoubling),
        ::testing::Values(0)));

INSTANTIATE_TEST_CASE_P(
    AllreducePcxKing,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          // TODO: Make Ring work with all possible sizes of context (especially for a context of size == 1).
          std::vector<int>({2, 4, 8, 16, 32})), // TODO: King currently support only context size which is power of 2 (meaning 1,2,4,8, etc.).
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreducePcxKing),
        ::testing::Values(0)));


INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase2,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 2, 4, 8, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(2)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase3,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(std::vector<int>({1, 3, 9, 27})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(3)));

INSTANTIATE_TEST_CASE_P(
    AllreduceBcubeBase4,
    AllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          std::vector<int>({1, 4, 16})),
        ::testing::ValuesIn(std::vector<int>({1, 64, 1000})),
        ::testing::Values(allreduceBcube),
        ::testing::Values(4)));

using Algorithm = AllreduceOptions::Algorithm;
using NewParam = std::tuple<int, int, int, bool, Algorithm>;

class AllreduceNewTest : public BaseTest,
                         public ::testing::WithParamInterface<NewParam> {};

TEST_P(AllreduceNewTest, Default) {
  auto contextSize = std::get<0>(GetParam());
  auto numPointers = std::get<1>(GetParam());
  auto dataSize = std::get<2>(GetParam());
  auto inPlace = std::get<3>(GetParam());
  auto algorithm = std::get<4>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    Fixture<uint64_t> inputs(context, numPointers, dataSize);
    Fixture<uint64_t> outputs(context, numPointers, dataSize);

    AllreduceOptions opts(context);
    opts.setAlgorithm(algorithm);
    opts.setOutputs(outputs.getPointers(), dataSize);
    if (inPlace) {
      outputs.assignValues();
    } else {
      opts.setInputs(inputs.getPointers(), dataSize);
      inputs.assignValues();
      outputs.clear();
    }

    opts.setReduceFunction([](void* a, const void* b, const void* c, size_t n) {
      auto ua = static_cast<uint64_t*>(a);
      const auto ub = static_cast<const uint64_t*>(b);
      const auto uc = static_cast<const uint64_t*>(c);
      for (size_t i = 0; i < n; i++) {
        ua[i] = ub[i] + uc[i];
      }
    });

    // A small maximum segment size triggers code paths where we'll
    // have a number of segments larger than the lower bound of
    // twice the context size.
    opts.setMaxSegmentSize(128);

    allreduce(opts);

    const auto stride = contextSize * numPointers;
    const auto base = (stride * (stride - 1)) / 2;
    const auto out = outputs.getPointers();
    for (auto j = 0; j < numPointers; j++) {
      for (auto k = 0; k < dataSize; k++) {
        ASSERT_EQ(k * stride * stride + base, out[j][k])
          << "Mismatch at out[" << j << "][" << k << "]";
      }
    }
  });
}

INSTANTIATE_TEST_CASE_P(
    AllreduceNewRing,
    AllreduceNewTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 7),
        ::testing::Values(1, 2, 3),
        ::testing::Values(1, 10, 100, 1000, 10000),
        ::testing::Values(true, false),
        ::testing::Values(Algorithm::RING)));

INSTANTIATE_TEST_CASE_P(
    AllreduceNewBcube,
    AllreduceNewTest,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 7),
        ::testing::Values(1, 2, 3),
        ::testing::Values(1, 10, 100, 1000, 10000),
        ::testing::Values(true, false),
        ::testing::Values(Algorithm::BCUBE)));

template <typename T>
AllreduceOptions::Func getFunction() {
  void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
  return AllreduceOptions::Func(func);
}

TEST_F(AllreduceNewTest, TestTimeout) {
  spawn(2, [&](std::shared_ptr<Context> context) {
    Fixture<uint64_t> outputs(context, 1, 1);
    AllreduceOptions opts(context);
    opts.setOutputs(outputs.getPointers(), 1);
    opts.setReduceFunction(getFunction<uint64_t>());
    opts.setTimeout(std::chrono::milliseconds(10));
    if (context->rank == 0) {
      try {
        allreduce(opts);
        FAIL() << "Expected exception to be thrown";
      } catch (::gloo::IoException& e) {
        ASSERT_NE(std::string(e.what()).find("Timed out"), std::string::npos);
      }
    }
  });
}


using RepeatParam = std::tuple<int, int, std::function<Func>, int, int>;
class RepeatAllreduceTest : public BaseTest,
                      public ::testing::WithParamInterface<RepeatParam> {};

TEST_P(RepeatAllreduceTest, RepeatSinglePointer) {
  auto contextSize = std::get<0>(GetParam());
  auto dataSize = std::get<1>(GetParam());
  auto fn = std::get<2>(GetParam());
  auto base = std::get<3>(GetParam());
  auto repeat = std::get<4>(GetParam());

  spawn(contextSize, [&](std::shared_ptr<Context> context) {
    const auto contextRank = context->rank;
    auto buffer = newBuffer<float>(dataSize);
    auto* ptr = buffer.data();
    for (int i = 0; i < dataSize; i++) {
      ptr[i] = contextRank;
    }

    fn(context, std::vector<float*>{ptr}, dataSize, repeat);

    auto ranks_sum = (contextSize * (contextSize - 1)) / 2;
    auto expected = ranks_sum * pow(contextSize, repeat - 1);
    for (int i = 0; i < dataSize; i++) {
      ASSERT_EQ(expected, ptr[i]) << "Mismatch at index " << i;
    }
  }, base);
}

INSTANTIATE_TEST_CASE_P(
    RepeatAllreducePcxKing,
    RepeatAllreduceTest,
    ::testing::Combine(
        ::testing::ValuesIn(
          // TODO: Make Ring work with all possible sizes of context (especially for a context of size == 1).
          std::vector<int>({2, 4, 8, 16})), // TODO: King currently support only context size which is power of 2 (meaning 1,2,4,8, etc.). // TODO: Need to add {32}. Currently it fails on data mismatch
        ::testing::ValuesIn(std::vector<int>({1, 64})), // TODO: Need to add {1000}. Currently it fails on data mismatch
        ::testing::Values(allreducePcxKing),
        ::testing::Values(0), // Base
        ::testing::ValuesIn(std::vector<int>({1,2})))); // Times to run the algorithm. // TODO: Need to add {3,4,5} times too. Currently it fails on data mismatch

INSTANTIATE_TEST_CASE_P(
    RepeatAllreducePcxRing,
    RepeatAllreduceTest,
    ::testing::Combine(
        // TODO: Make Ring work with all possible sizes of context (especially for a context of size == 1).
        ::testing::ValuesIn(std::vector<int>({2,4,8})), //::testing::Range(2, 16,1), // Start, End, Step size // TODO: Need to odd numbers of context size and also some large numbers like 16,32...
                                                        // context size equals to 6 fails on data mismatch
                                                        // For supporting context size equals to 1 need to add "return" in the run()
                                                        // For supporting odd context size need to change the way QPs exchange information out of band. 
        ::testing::ValuesIn(std::vector<int>({8,16,32})), // TODO: Need to add {64,1000}. Currently it fails on data mismatch // TODO: Need to add very small sizes like 1,2... Fails probably because of some integer devision that zerofies the number of bytes to reduce. // TODO: This parameter should be a multiplier of the context size parameter. Meaning cotnext size x can support only x,2x,3x etc. count size.
        ::testing::Values(allreducePcxRing),
        ::testing::Values(0), // Base
        ::testing::ValuesIn(std::vector<int>({1,2,3,4,5})))); // Times to run the algorithm // TODO: Need to add some large number of repeats

} // namespace
} // namespace test
} // namespace gloo
