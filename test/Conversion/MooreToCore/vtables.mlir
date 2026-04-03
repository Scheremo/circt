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

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

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
