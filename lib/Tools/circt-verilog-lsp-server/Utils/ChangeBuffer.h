//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CHANGEBUFFER_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_CHANGEBUFFER_H_

#include "llvm/ADT/ArrayRef.h"

namespace circt {
namespace lsp {

/// Thread-safe buffer for accumulating `EventT`s.
template <typename EventT>
struct ChangeBuffer {
  /// Add one batch (from one didChange). Assumes versions are non-decreasing.
  /// Thread-safe.
  void add(llvm::ArrayRef<EventT> changes, int64_t version) {
    std::lock_guard<std::mutex> lk(mu);
    // Sometimes a change is the full replacement of a file; discard previous
    // changes in such a case.
    if (version < pendingVersion)
      return; // ignore stale batch
    pending.insert(pending.end(), changes.begin(), changes.end());
    pendingVersion = version;
  }

  /// Drain all buffered changes atomically. Thread-safe.
  std::vector<EventT> drain(int64_t &versionOut) {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<EventT> out;
    out.swap(pending);
    versionOut = pendingVersion;
    return out;
  }

  /// Snapshot-only (no drain). Thread-safe.
  bool empty() const {
    std::lock_guard<std::mutex> lk(mu);
    return pending.empty();
  }

private:
  mutable std::mutex mu;
  std::vector<EventT> pending; // buffered incremental changes (in order)
  int64_t pendingVersion = 0;  // last seen version for these changes
};
} // namespace lsp
} // namespace circt

#endif
