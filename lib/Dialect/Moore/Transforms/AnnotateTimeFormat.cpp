//===- AnnotateTimeformat.cpp -----------------------------------*- C++ -*-===//
//
// Test-only pass: resolve the effective $timeformat at each operation and
// annotate it as `resolved_timeformat = #moore.timeformat<...>`.
//
// Usage in lit:
//   // RUN: circt-opt %s -moore-annotate-timeformat | FileCheck %s
//
// This pass is intentionally simple (linear scan per block) and meant for
// tests. If you later need dominance-aware behavior, you can expand the logic.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreAttributes.h" // TimeFormatAttr
#include "circt/Dialect/Moore/MooreOps.h"        // TimeformatBIOp
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include <optional>
#include <string>

namespace circt::moore {
namespace {

// Thread/annotate through one region; recurse into nested regions.
static void annotateRegion(mlir::Region &region,
                           const TimeFormatAttr &incoming) {
  for (mlir::Block &block : region) {
    TimeFormatAttr cur = incoming;

    for (mlir::Operation &oref : block) {
      mlir::Operation *op = &oref;
      mlir::MLIRContext *ctx = op->getContext();

      // 1) Attach the currently resolved state at this op.
      op->setAttr("resolved_timeformat", cur);

      // 2) If this is a writer, update the running state for following ops.
      if (auto tf = llvm::dyn_cast<TimeformatBIOp>(op)) {
        auto units_number = tf.getUnitsNumber() ? tf.getUnitsNumber().value()
                                                : cur.getUnitsNumber();
        auto precision_number = tf.getPrecisionNumber()
                                    ? tf.getPrecisionNumber().value()
                                    : cur.getPrecisionNumber();
        auto string_suffix = tf.getSuffixString() ? tf.getSuffixStringAttr()
                                                  : cur.getSuffixString();
        auto minimum_field_width = tf.getMinimumFieldWidth()
                                       ? tf.getMinimumFieldWidth().value()
                                       : cur.getMinimumFieldWidth();
        cur = TimeFormatAttr::get(ctx, units_number, precision_number,
                                  string_suffix, minimum_field_width);
      }

      // 3) Recurse into nested regions with the *current* state.
      for (mlir::Region &nested : op->getRegions())
        annotateRegion(nested, cur);
    }
  }
}

struct AnnotateTimeformatPass
    : mlir::PassWrapper<AnnotateTimeformatPass,
                        mlir::OperationPass<moore::SVModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnnotateTimeformatPass)

  llvm::StringRef getArgument() const final {
    return "moore-annotate-timeformat";
  }
  llvm::StringRef getDescription() const final {
    return "Annotate each operation with the resolved $timeformat state";
  }

  void runOnOperation() override {
    moore::SVModuleOp mod = this->getOperation();

    // Start each top-level region with default state. If you want the defaults
    // to come from the module's timeunit/timeprecision, plumb that here.
    auto *ctx = mod->getContext();

auto defaults = TimeFormatAttr::get(
    ctx, int8_t(-15), int8_t(-15), mlir::StringAttr::get(ctx, ""), uint8_t(20));

// Cast to the base (or use llvm::isa free function).
mlir::Attribute a = defaults;

llvm::errs() << "isa TimeFormatAttr? "
             << llvm::isa<TimeFormatAttr>(a) << "\n";
llvm::errs() << "dialect=" << a.getDialect().getNamespace() << "\n";

// Safe field access via concrete getters:
llvm::errs() << "units=" << int(defaults.getUnitsNumber())
             << " prec=" << int(defaults.getPrecisionNumber())
             << " width=" << int(defaults.getMinimumFieldWidth())
             << " suffix=`" << defaults.getSuffixString().getValue() << "`\n";

// Print the attr itself:
llvm::errs() << a << "\n";


    for (mlir::Region &region : mod->getRegions())
      annotateRegion(region, defaults);
  }
};

} // namespace

// Factory function (reference this in Passes.td)
std::unique_ptr<mlir::Pass> createAnnotateTimeformatPass() {
  return std::make_unique<AnnotateTimeformatPass>();
}

} // namespace circt::moore
