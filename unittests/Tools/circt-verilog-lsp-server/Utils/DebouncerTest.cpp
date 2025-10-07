#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <thread>

#include "Debouncer.h"

using namespace circt::lsp;
using namespace std::chrono;

namespace {

// Small helper to wait on a predicate with a timeout.
template <typename Pred>
bool waitUntil(Pred pred, milliseconds timeout,
               milliseconds poll = milliseconds(1)) {
  const auto start = steady_clock::now();
  while (steady_clock::now() - start < timeout) {
    if (pred())
      return true;
    std::this_thread::sleep_for(poll);
  }
  return pred();
}

TEST(DebouncerTest, RunsOnceAfterQuietPeriod) {
  Debouncer d(50ms); // trailing-edge delay
  std::atomic<int> ran{0};

  // Spam schedule() many times within the delay window.
  for (int i = 0; i < 10; ++i) {
    d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
    std::this_thread::sleep_for(5ms);
  }

  // Should fire exactly once after quiet period.
  ASSERT_TRUE(waitUntil([&] { return ran.load() == 1; }, 500ms));
  EXPECT_EQ(ran.load(), 1);
}

TEST(DebouncerTest, ResetsQuietPeriodOnReschedule) {
  Debouncer d(100ms);
  std::atomic<int> ran{0};

  d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
  // Re-schedule halfway through the quiet window; should push execution out.
  std::this_thread::sleep_for(50ms);
  d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });

  // Before 100ms from the last schedule, it should not have fired.
  std::this_thread::sleep_for(60ms);
  EXPECT_EQ(ran.load(), 0);

  // Then it should fire once.
  ASSERT_TRUE(waitUntil([&] { return ran.load() == 1; }, 500ms));
  EXPECT_EQ(ran.load(), 1);
}

TEST(DebouncerTest, MaxWaitCapsTheDelay) {
  // delay=100ms but maxWait=150ms means repeated schedules won't push firing
  // beyond 150ms.
  Debouncer d(100ms, 150ms);
  std::atomic<int> ran{0};

  auto scheduleBurst = [&] {
    d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
  };

  // Keep re-scheduling within the delay window.
  auto start = steady_clock::now();
  while (steady_clock::now() - start < 140ms) {
    scheduleBurst();
    std::this_thread::sleep_for(30ms);
  }

  ASSERT_TRUE(waitUntil([&] { return ran.load() >= 1; }, 400ms));
  EXPECT_EQ(ran.load(), 1); // only one trailing-edge fire
}

TEST(DebouncerTest, FlushRunsImmediately) {
  Debouncer d(2s); // big delay to prove flush works
  std::atomic<int> ran{0};

  d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
  d.flush(); // should trigger without waiting 2s
  ASSERT_TRUE(waitUntil([&] { return ran.load() == 1; }, 200ms));
}

TEST(DebouncerTest, CancelPreventsRun) {
  Debouncer d(50ms);
  std::atomic<int> ran{0};

  d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
  d.cancel();

  std::this_thread::sleep_for(150ms);
  EXPECT_EQ(ran.load(), 0);
}

TEST(DebouncerTest, DestructorCancelsPendingCallback) {
  std::atomic<int> ran{0};
  {
    Debouncer d(200ms);
    d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
    // d goes out of scope and should cancel the pending callback.
  }
  std::this_thread::sleep_for(300ms);
  EXPECT_EQ(ran.load(), 0);
}

// Ensure thread safety under concurrent schedules: only one run after quiet
// period.
TEST(DebouncerTest, ConcurrentSchedulesCollapseToOneRun) {
  Debouncer d(80ms);
  std::atomic<int> ran{0};

  auto worker = [&] {
    for (int i = 0; i < 10; ++i) {
      d.schedule([&] { ran.fetch_add(1, std::memory_order_relaxed); });
      std::this_thread::sleep_for(5ms);
    }
  };

  std::thread t1(worker), t2(worker), t3(worker);
  t1.join();
  t2.join();
  t3.join();

  ASSERT_TRUE(waitUntil([&] { return ran.load() >= 1; }, 500ms));
  EXPECT_EQ(ran.load(), 1);
}

} // namespace
