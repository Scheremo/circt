// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

/// Check that a classdecl gets translated to a llvm struct type

// CHECK-LABEL: module {
// CHECK: llvm.mlir.global internal @__type_PropertyCombo() {addr_space = 0 : i32}
// CHECK-SAME: : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)> {
// CHECK: [[U:%.*]] = llvm.mlir.undef : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
// CHECK: llvm.return [[U]] : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
// CHECK: }
// CHECK: }

// (Optional sanity: the class decl should be gone after conversion.)
// CHECK-NOT: moore.class.classdecl

module {
  moore.class.classdecl @PropertyCombo : {
    moore.class.propertydecl @pubAutoI32   : !moore.i32
    moore.class.propertydecl @protStatL18  : !moore.l18
    moore.class.propertydecl @localAutoI32 : !moore.i32
  }

  llvm.mlir.global internal @__type_PropertyCombo()
      : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)> {
    %u = llvm.mlir.undef
         : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
    llvm.return %u
         : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
  }
}
