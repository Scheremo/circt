#include "gtest/gtest.h"

#include <atomic>
#include <future>
#include <thread>

#include "DocChangeBucket.h"

using namespace circt::lsp;
using namespace std::chrono;

namespace {

TEST(DocChangeBucketTest, ScheduleRunsWithoutDebounce) {
  auto b = std::make_shared<DocChangeBucket<int>>();

  std::promise<void> ran;
  std::vector<int> seen;
  int64_t seenVersion = -1;

  b->enqueueChanges({1, 2, 3}, 7);
  b->scheduleUpdate([&](const std::vector<int> &batch, int64_t v) {
    seen = batch;
    seenVersion = v;
    ran.set_value();
  });

  ASSERT_EQ(ran.get_future().wait_for(200ms), std::future_status::ready);
  EXPECT_EQ(seen.size(), 3u);
  EXPECT_EQ(seenVersion, 7);
}

TEST(DocChangeBucketTest, DebounceCoalescesTrailing) {
  auto b = std::make_shared<DocChangeBucket<int>>(/*useDebounce=*/true,
                                                  /*minMs=*/20, /*maxMs=*/0);

  std::promise<void> ran;
  std::atomic<int> calls{0};
  std::atomic<int64_t> lastVersion{-1};

  // Rapidly enqueue/schedule a few times; only the last should run.
  for (int i = 1; i <= 3; ++i) {
    b->enqueueChanges({i}, i);
    b->scheduleUpdate([&](const std::vector<int> &batch, int64_t v) {
      calls.fetch_add(1, std::memory_order_relaxed);
      lastVersion.store(v, std::memory_order_relaxed);
      ran.set_value();
    });
    std::this_thread::sleep_for(5ms); // << less than debounce delay
  }

  ASSERT_EQ(ran.get_future().wait_for(500ms), std::future_status::ready);
  // Allow a brief tail for extra callbacks (shouldn't be any).
  std::this_thread::sleep_for(30ms);

  EXPECT_EQ(calls.load(), 1);
  EXPECT_EQ(lastVersion.load(), 3);
}

TEST(DocChangeBucketTest, CancelUpdatePreventsRun) {
  auto b = std::make_shared<DocChangeBucket<int>>(/*useDebounce=*/true,
                                                  /*minMs=*/50, /*maxMs=*/0);

  std::promise<void> ran;
  bool executed = false;

  b->enqueueChanges({42}, 9);
  b->scheduleUpdate([&](const std::vector<int> &, int64_t) {
    executed = true;
    ran.set_value();
  });

  // Cancel before debounce delay elapses.
  b->cancelUpdate();

  // Give ample time; should not run.
  EXPECT_NE(ran.get_future().wait_for(120ms), std::future_status::ready);
  EXPECT_FALSE(executed);
}

TEST(DocChangeBucketTest, EmptyQueueDoesNotRun) {
  auto b = std::make_shared<DocChangeBucket<int>>();
  std::atomic<bool> ran{false};

  b->scheduleUpdate(
      [&](const std::vector<int> &, int64_t) { ran.store(true); });

  // No enqueueChanges, so nothing should run.
  std::this_thread::sleep_for(20ms);
  EXPECT_FALSE(ran.load());
}

} // namespace
