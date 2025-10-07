//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_DOCCHANGEBUCKET_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_DOCCHANGEBUCKET_H_

#include "ChangeBuffer.h"
#include "Debouncer.h"
#include "llvm/ADT/StringMap.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace circt {
namespace lsp {

template <typename T>
struct DocChangeBucket : std::enable_shared_from_this<DocChangeBucket<T>> {
  /// A function that accepts a vector of `T` and a version number (int64_t)
  using StepFn = std::function<void(const std::vector<T> &, int64_t)>;

  DocChangeBucket(bool useDb, uint32_t minMs, uint32_t maxMs)
      : useDebounce(useDb) {
    if (useDebounce) {
      std::optional<std::chrono::milliseconds> maxDelay;
      if (maxMs > 0)
        maxDelay = std::chrono::milliseconds(maxMs);

      deb = std::make_unique<Debouncer>(std::chrono::milliseconds(minMs),
                                        maxDelay);
    }
  }

  DocChangeBucket() : useDebounce(false), deb(nullptr) {}

  /// Thread-safe function to enqueue new changes
  void enqueueChanges(const std::vector<T> &update, int64_t version) {
    changeBuf.add(update, version);
  }

  /// Thread-safe function to cancel any outstanding updates
  void cancelUpdate() {
    if (deb)
      deb->cancel();
  }

  /// Thread-safe function to enqueue new update; performs all necessary checks
  /// before applying
  template <typename Fn>
  void scheduleUpdate(Fn &&step) {
    auto weak = this->weak_from_this();
    auto stepFn = StepFn(std::forward<Fn>(step));

    // Fallback-friendly worker: runs even if not owned by shared_ptr
    auto work = [weak, raw = this, stepFn = std::move(stepFn)]() mutable {
      // Prefer shared ownership; fall back to raw if not managed.
      auto fb = weak.lock();
      auto *p = fb ? fb.get() : raw;

      // Respect liveness flag (works for both paths).
      if (!p->alive.load(std::memory_order_acquire))
        return;

      // Build gate; if busy, ask for rerun and bail.
      if (p->building.exchange(true, std::memory_order_acq_rel)) {
        p->rerun.store(true, std::memory_order_release);
        return;
      }

      // Loop while rerun requested; always drain first to avoid races.
      do {
        p->rerun.store(false, std::memory_order_release);

        int64_t version = 0;
        auto batch = p->changeBuf.drain(version);
        if (batch.empty()) // nothing to do; drop gate and exit
          break;

        stepFn(batch, version);

      } while (p->rerun.load(std::memory_order_acquire));

      p->building.store(false, std::memory_order_release);
    };

    if (useDebounce && deb)
      deb->schedule(std::move(work));
    else
      work();
  }

  /// Atomic variable indicating whether the underlying file is alive
  std::atomic<bool> alive{true};
  const bool useDebounce;

private:
  std::atomic<bool> building{false}; // build gate
  std::atomic<bool> rerun{false};    // “build again” flag

  ChangeBuffer<T> changeBuf;      // batched edits
  std::unique_ptr<Debouncer> deb; // trailing-edge debounce
};

// C++17-style detection for presence of .alive
template <typename U, typename = void>
struct has_alive_member : std::false_type {};
template <typename U>
struct has_alive_member<U, std::void_t<decltype(std::declval<U &>().alive)>>
    : std::true_type {};

// C++17-style detection for presence of .deb
template <typename U, typename = void>
struct has_deb_member : std::false_type {};
template <typename U>
struct has_deb_member<U, std::void_t<decltype(std::declval<U &>().deb)>>
    : std::true_type {};

template <typename BucketT>
class BucketRegistry {
public:
  using BucketPtr = std::shared_ptr<BucketT>;
  using FactoryFn = std::function<BucketPtr()>;
  using CancelFn = std::function<void(BucketT &)>;

  // Construct with a factory for new buckets and an optional custom cancel fn.
  // If no cancel fn is supplied, a default is used (alive=false;
  // deb->cancel()).
  explicit BucketRegistry(FactoryFn factory, CancelFn cancel = defaultCancel)
      : factoryFn(std::move(factory)), cancelFn(std::move(cancel)) {}

  // Thread-safe create/find-or-return
  BucketPtr getOrCreate(llvm::StringRef key) {
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = map.find(key); it != map.end())
      return it->second;
    BucketPtr fb = factoryFn();
    map.try_emplace(key, fb);
    return fb;
  }

  // Thread-safe find (nullptr if missing)
  BucketPtr find(llvm::StringRef key) const {
    std::lock_guard<std::mutex> lk(mu);
    auto it = map.find(key);
    return (it == map.end()) ? nullptr : it->second;
  }

  // Thread-safe erase of a single bucket (also cancels it)
  void erase(llvm::StringRef key) {
    BucketPtr fb;
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = map.find(key);
      if (it == map.end())
        return;
      fb = it->second; // keep alive outside lock
      map.erase(it);
    }
    safeCancel(fb);
  }

  // Thread-safe cancel + clear of all buckets
  void cancelAll() {
    std::vector<BucketPtr> snapshot;
    {
      std::lock_guard<std::mutex> lk(mu);
      snapshot.reserve(map.size());
      for (auto &kv : map)
        snapshot.push_back(kv.second);
      map.clear();
    }
    for (auto &fb : snapshot)
      safeCancel(fb);
  }

  // Optional helpers
  size_t size() const {
    std::lock_guard<std::mutex> lk(mu);
    return map.size();
  }

  bool empty() const { return size() == 0; }

private:
  static void defaultCancel(BucketT &b) {
    if constexpr (has_alive_member<BucketT>::value) {
      b.alive.store(false, std::memory_order_release);
    }
    if constexpr (has_deb_member<BucketT>::value) {
      if (b.deb)
        b.deb->cancel();
    }
  }

  void safeCancel(const BucketPtr &fb) {
    if (fb)
      cancelFn(*fb);
  }

  FactoryFn factoryFn;
  CancelFn cancelFn;

  mutable std::mutex mu;
  llvm::StringMap<BucketPtr> map;
};

} // namespace lsp
} // namespace circt

#endif
