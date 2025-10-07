
#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_DEBOUNCER_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_DEBOUNCER_H_

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>

namespace circt {
namespace lsp {

/// Thread-safe trailing-edge scheduler for delaying `onDidChange` events
struct Debouncer {
public:
  explicit Debouncer(
      std::chrono::steady_clock::duration delay,
      std::optional<std::chrono::steady_clock::duration> maxWait = std::nullopt)
      : delay(delay), maxWait(maxWait), worker(&Debouncer::run, this) {}

  ~Debouncer() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      stop = true;
      ++version;
    }
    cv.notify_all();
    if (worker.joinable())
      worker.join();
  }

  /// Schedule fn to run after delay. Replaces any pending callback.
  /// Thread-safe.
  void schedule(std::function<void()> fn) {
    std::lock_guard<std::mutex> lk(mutex);
    callback = std::move(fn);
    lastSchedule = std::chrono::steady_clock::now();
    if (!firstSchedule)
      firstSchedule = *lastSchedule;
    ++version;
    cv.notify_all();
  }

  /// Force the pending callback to run immediately. Thread-safe.
  void flush() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      if (!callback)
        return;
      lastSchedule = std::chrono::steady_clock::now() - delay; // backdate
      ++version;
    }
    cv.notify_all();
  }

  /// Cancel the pending callback, if any. Thread-safe.
  void cancel() {
    std::lock_guard<std::mutex> lk(mutex);
    callback.reset();
    firstSchedule.reset();
    ++version;
    cv.notify_all();
  }

private:
  void run() {
    std::unique_lock<std::mutex> lk(mutex);
    while (!stop) {
      if (!callback) {
        cv.wait(lk, [&] { return stop || callback.has_value(); });
        if (stop)
          break;
      }

      assert(lastSchedule.has_value() && "Debouncer invariant violated");

      auto v = version.load(std::memory_order_acquire);
      auto now = std::chrono::steady_clock::now();
      auto nextTime = *lastSchedule + delay;

      if (maxWait && firstSchedule) {
        auto maxTime = *firstSchedule + *maxWait;
        if (maxTime < nextTime)
          nextTime = maxTime;
      }

      if (now < nextTime) {
        cv.wait_until(lk, nextTime, [&] {
          return stop || version.load(std::memory_order_acquire) != v ||
                 !callback.has_value();
        });
        continue;
      }

      auto fn = std::move(*callback);
      callback.reset();
      firstSchedule.reset();
      lk.unlock();
      std::thread(std::move(fn))
          .detach(); // fn might take a long time, so offload and re-arm timer.
      lk.lock();
    }
  }

  const std::chrono::steady_clock::duration delay;
  const std::optional<std::chrono::steady_clock::duration> maxWait;

  std::mutex mutex;
  std::condition_variable cv;
  std::optional<std::chrono::steady_clock::time_point> firstSchedule;
  std::optional<std::chrono::steady_clock::time_point> lastSchedule;
  std::optional<std::function<void()>> callback;
  std::atomic<uint64_t> version{0};
  bool stop = false;

  std::thread worker;
};
} // namespace lsp
} // namespace circt

#endif
