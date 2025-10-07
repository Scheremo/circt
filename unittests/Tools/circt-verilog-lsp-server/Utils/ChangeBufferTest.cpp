#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "ChangeBuffer.h"

using namespace circt::lsp;
using namespace std::chrono;

namespace {

TEST(ChangeBufferTest, StartsEmpty) {
  ChangeBuffer<int> buf;
  EXPECT_TRUE(buf.empty());
  int64_t v = -1;
  auto out = buf.drain(v);
  EXPECT_TRUE(out.empty());
  EXPECT_EQ(v, 0); // default-initialized pendingVersion
}

TEST(ChangeBufferTest, AddAndDrainSingleBatch) {
  ChangeBuffer<int> buf;

  std::vector<int> batch{1, 2, 3};
  buf.add(batch, /*version=*/5);

  EXPECT_FALSE(buf.empty());

  int64_t v = -1;
  auto out = buf.drain(v);
  EXPECT_EQ(v, 5);
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
  EXPECT_EQ(out[2], 3);

  EXPECT_TRUE(buf.empty()); // drained
}

TEST(ChangeBufferTest, AppendMultipleBatchesAndPreserveOrderWithinBatch) {
  ChangeBuffer<int> buf;

  std::vector<int> a{10, 11};
  std::vector<int> b{20, 21, 22};

  buf.add(a, /*version=*/1);
  buf.add(b, /*version=*/2);

  int64_t v = -1;
  auto out = buf.drain(v);
  EXPECT_EQ(v, 2);
  ASSERT_EQ(out.size(), 5u);

  // Order is a then b (within-batch order must be preserved).
  std::vector<int> expect{10, 11, 20, 21, 22};
  EXPECT_EQ(out, expect);
}

TEST(ChangeBufferTest, StaleBatchIsIgnored) {
  ChangeBuffer<int> buf;

  std::vector<int> first{1, 2};
  std::vector<int> stale{9, 9};

  buf.add(first, /*version=*/7);
  // Lower version should be ignored completely.
  buf.add(stale, /*version=*/6);

  int64_t v = -1;
  auto out = buf.drain(v);
  EXPECT_EQ(v, 7);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], 1);
  EXPECT_EQ(out[1], 2);
}

TEST(ChangeBufferTest, EqualVersionIsAcceptedAndAppended) {
  ChangeBuffer<int> buf;

  std::vector<int> a{1};
  std::vector<int> b{2, 3};

  buf.add(a, /*version=*/4);
  buf.add(b, /*version=*/4); // equal version appends per implementation

  int64_t v = -1;
  auto out = buf.drain(v);
  EXPECT_EQ(v, 4);
  std::vector<int> expect{1, 2, 3};
  EXPECT_EQ(out, expect);
}

TEST(ChangeBufferTest, DrainClearsAndSubsequentAddsWork) {
  ChangeBuffer<int> buf;

  buf.add(std::vector<int>{1}, /*version=*/1);
  int64_t v = -1;
  auto out1 = buf.drain(v);
  EXPECT_EQ(v, 1);
  EXPECT_TRUE(buf.empty());

  buf.add(std::vector<int>{5, 6}, /*version=*/3);
  auto out2 = buf.drain(v);
  EXPECT_EQ(v, 3);
  std::vector<int> expect{5, 6};
  EXPECT_EQ(out2, expect);
}

} // namespace
