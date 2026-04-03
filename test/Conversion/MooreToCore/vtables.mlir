// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-LABEL: llvm.mlir.global internal constant @"tClass::vtable"()
// CHECK-SAME: : !llvm.struct<(ptr, ptr)> {
// CHECK: [[OVR:%.*]] = llvm.mlir.addressof @"tClass::subroutine" : !llvm.ptr
// CHECK: llvm.insertvalue [[OVR]]
// CHECK: [[BASE:%.*]] = llvm.mlir.addressof @"testClass::testSubroutine" : !llvm.ptr
// CHECK: llvm.insertvalue [[BASE]]
// CHECK: llvm.return

// CHECK-LABEL: llvm.mlir.global internal constant @"testClass::vtable"()
// CHECK-SAME: : !llvm.struct<(ptr, ptr)> {
// CHECK: [[SUB:%.*]] = llvm.mlir.addressof @"testClass::subroutine" : !llvm.ptr
// CHECK: llvm.insertvalue [[SUB]]
// CHECK: [[VT:%.*]] = llvm.mlir.addressof @"testClass::testSubroutine" : !llvm.ptr
// CHECK: llvm.insertvalue [[VT]]
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @"testClass::subroutine"(
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @"testClass::testSubroutine"(
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @"tClass::subroutine"(
// CHECK: llvm.return

// CHECK-LABEL: func.func private @alloc_test()
// CHECK: %[[SIZE:.*]] = llvm.mlir.constant
// CHECK: %[[OBJ:.*]] = call @malloc(%[[SIZE]]) : (i64) -> !llvm.ptr
// CHECK: %[[RTTI:.*]] = llvm.mlir.addressof @"testClass::typeinfo" : !llvm.ptr
// CHECK: %[[HEADER_PTR:.*]] = llvm.getelementptr %[[OBJ]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClass"
// CHECK: %[[TYPEINFO_PTR:.*]] = llvm.getelementptr %[[HEADER_PTR]][%{{.*}}, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: llvm.store %[[RTTI]], %[[TYPEINFO_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: %[[VTBL:.*]] = llvm.mlir.addressof @"testClass::vtable" : !llvm.ptr
// CHECK: %[[VTBL_PTR:.*]] = llvm.getelementptr %[[HEADER_PTR]][%{{.*}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: llvm.store %[[VTBL]], %[[VTBL_PTR]] : !llvm.ptr, !llvm.ptr
// CHECK: return

// CHECK-LABEL: func.func private @dispatch_test(
// CHECK-SAME: %[[OBJ:.*]]: !llvm.ptr
// CHECK: %[[HEADER_PTR2:.*]] = llvm.getelementptr %[[OBJ]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClass"
// CHECK: %[[VTBL_PTR_PTR:.*]] = llvm.getelementptr %[[HEADER_PTR2]][%{{.*}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: %[[VTBL_PTR:.*]] = llvm.load %[[VTBL_PTR_PTR]] : !llvm.ptr
// CHECK: %[[SLOT_PTR:.*]] = llvm.getelementptr %[[VTBL_PTR]][%{{.*}}, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: %[[METH_PTR:.*]] = llvm.load %[[SLOT_PTR]] : !llvm.ptr
// CHECK: llvm.call %[[METH_PTR]](%[[OBJ]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK: return

// CHECK-LABEL: func.func private @dispatch_upcast_test(
// CHECK-SAME: %[[DERIVED:.*]]: !llvm.ptr
// CHECK-NOT: moore.class.upcast
// CHECK: %[[HEADER_PTR3:.*]] = llvm.getelementptr %[[DERIVED]][%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"testClass"
// CHECK: %[[VTBL_PTR_PTR2:.*]] = llvm.getelementptr %[[HEADER_PTR3]][%{{.*}}, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: %[[VTBL_PTR2:.*]] = llvm.load %[[VTBL_PTR_PTR2]] : !llvm.ptr
// CHECK: %[[SLOT_PTR2:.*]] = llvm.getelementptr %[[VTBL_PTR2]][%{{.*}}, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK: %[[METH_PTR2:.*]] = llvm.load %[[SLOT_PTR2]] : !llvm.ptr
// CHECK: llvm.call %[[METH_PTR2]](%[[DERIVED]]) : !llvm.ptr, (!llvm.ptr) -> ()
// CHECK: return

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry
// CHECK-NOT: moore.vtable.load_method

moore.class.classdecl @virtualFunctionClass {
  moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>) -> ()
}
moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
  moore.class.methoddecl @testSubroutine : (!moore.class<@realFunctionClass>) -> ()
}
moore.class.classdecl @testClass implements [@realFunctionClass] {
  moore.class.methoddecl @subroutine -> @"testClass::subroutine" : (!moore.class<@testClass>) -> ()
  moore.class.methoddecl @testSubroutine -> @"testClass::testSubroutine" : (!moore.class<@testClass>) -> ()
}
moore.vtable @testClass::@vtable {
  moore.vtable @realFunctionClass::@vtable {
    moore.vtable @virtualFunctionClass::@vtable {
      moore.vtable_entry @subroutine -> @"testClass::subroutine"
    }
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  moore.vtable_entry @subroutine -> @"testClass::subroutine"
  moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
}
func.func private @"testClass::subroutine"(%arg0: !moore.class<@testClass>) {
  return
}
func.func private @"testClass::testSubroutine"(%arg0: !moore.class<@testClass>) {
  return
}

moore.class.classdecl @tClass extends @testClass {
  moore.class.methoddecl @subroutine -> @"tClass::subroutine" : (!moore.class<@tClass>) -> ()
}
moore.vtable @tClass::@vtable {
  moore.vtable @testClass::@vtable {
    moore.vtable @realFunctionClass::@vtable {
      moore.vtable @virtualFunctionClass::@vtable {
        moore.vtable_entry @subroutine -> @"tClass::subroutine"
      }
      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
    }
    moore.vtable_entry @subroutine -> @"tClass::subroutine"
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  moore.vtable_entry @subroutine -> @"tClass::subroutine"
}
func.func private @"tClass::subroutine"(%arg0: !moore.class<@tClass>) {
  return
}

func.func private @alloc_test() {
  %0 = moore.class.new : <@testClass>
  return
}

func.func private @dispatch_test(%arg0: !moore.class<@testClass>) {
  %0 = moore.vtable.load_method %arg0 : @subroutine of <@testClass> -> (!moore.class<@testClass>) -> ()
  call_indirect %0(%arg0) : (!moore.class<@testClass>) -> ()
  return
}

func.func private @dispatch_upcast_test(%arg0: !moore.class<@tClass>) {
  %0 = moore.class.upcast %arg0 : <@tClass> to <@testClass>
  %1 = moore.vtable.load_method %0 : @subroutine of <@testClass> -> (!moore.class<@testClass>) -> ()
  call_indirect %1(%0) : (!moore.class<@testClass>) -> ()
  return
}
