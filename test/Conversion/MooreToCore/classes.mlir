// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

/// Sanity check that a classdecl gets translated to a llvm struct type

// CHECK-LABEL: module {
// CHECK: llvm.mlir.global internal @__type_PropertyCombo() {addr_space = 0 : i32}
// CHECK-SAME: : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)> {
// CHECK: [[U:%.*]] = llvm.mlir.undef : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
// CHECK: llvm.return [[U]] : !llvm.struct<"moore.class.PropertyCombo", (i32, i18, i32)>
// CHECK: }
// CHECK: }

// Class decl needs to be gone.
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

/// Check that new lowers to malloc

// malloc should be declared in the LLVM dialect (opaque pointer mode).
// CHECK: llvm.func @malloc(i64) -> !llvm.ptr
// CHECK: func.func private @test_new
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(12 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// Original op must be gone.
// CHECK-NOT: moore.class.new

// Class decl should be erased by the class lowering pass.
// CHECK-NOT: moore.class.classdecl

module {
  // Minimal class so the identified struct has a concrete body.
  moore.class.classdecl @C : {
    moore.class.propertydecl @a : !moore.i32
    moore.class.propertydecl @b : !moore.l32
    moore.class.propertydecl @c : !moore.l32
  }

  // Allocate a new instance; should lower to llvm.call @malloc(i64).
  func.func private @test_new() {
    %h = moore.class.new : <@C>
    return
  }
}

/// Check that upcast lowers to nop

// CHECK-LABEL: func.func @f(
// CHECK-SAME: %[[ARG:.*]]: !llhd.ref<!llvm.ptr>) -> !llhd.ref<!llvm.ptr> {
// CHECK-NEXT:   return %[[ARG]] : !llhd.ref<!llvm.ptr>
// CHECK-NEXT: }

// CHECK-NOT: moore.class.upcast
// CHECK-NOT: moore.class.classdecl

module {
  // Minimal hierarchy so the upcast is well-typed pre-conversion.
  moore.class.classdecl @B : { }
  moore.class.classdecl @D extends @B : { }

  // Upcast from D to B.
  func.func @f(%r: !moore.ref<!moore.class.object<@D>>)
              -> !moore.ref<!moore.class.object<@B>> {
    %u = moore.class.upcast %r
         : <class.object<@D>> to <class.object<@B>>
    return %u : !moore.ref<!moore.class.object<@B>>
  }
}

/// Check that property_ref lowers to GEP

// CHECK-LABEL: func.func @get_a(
// CHECK-SAME: %arg0: !llhd.ref<!llvm.ptr>) -> i32 {
// CHECK: %[[OBJ:.*]] = llhd.prb %arg0 : !llvm.ptr
// CHECK: %[[C0A:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[C0B:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[OBJ]][%[[C0A]], 0]
// CHECK-SAME: : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"moore.class.CHere", (i32)>
// CHECK: %[[REF:.*]] = builtin.unrealized_conversion_cast %[[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK: %[[VAL:.*]] = llhd.prb %[[REF]] : i32
// CHECK: return %[[VAL]] : i32

// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.classdecl

module {
  moore.class.classdecl @CHere : {
    moore.class.propertydecl @a : !moore.i32
  }

  // Take &C::a and load it.
  func.func @get_a(%this: !moore.ref<!moore.class.object<@CHere>>) -> !moore.i32 {
    %p = moore.class.property_ref %this[@a] : <class.object<@CHere>> -> <i32>
    %v = moore.read %p : <i32>
    return %v : !moore.i32
  }
}
