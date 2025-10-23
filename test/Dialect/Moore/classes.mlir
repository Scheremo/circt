// RUN: circt-opt --verify-diagnostics --verify-roundtrip %s | FileCheck %s

module {
// CHECK-LABEL: moore.class.classdecl @Plain : {
// CHECK: }
moore.class.classdecl @Plain : {
}

// CHECK-LABEL:   moore.class.classdecl @I : {
// CHECK:   }
moore.class.classdecl @I : {
}

// CHECK-LABEL:   moore.class.classdecl @Base : {
// CHECK:   }
// CHECK:   moore.class.classdecl @Derived extends @Base : {
// CHECK:   }
moore.class.classdecl @Base : {
}
moore.class.classdecl @Derived extends @Base : {
}

// CHECK-LABEL:   moore.class.classdecl @IBase : {
// CHECK:   }
// CHECK:   moore.class.classdecl @IExt extends @IBase : {
// CHECK:   }

moore.class.classdecl @IBase : {
}
moore.class.classdecl @IExt extends @IBase : {
}

// CHECK-LABEL:   moore.class.classdecl @IU : {
// CHECK:   }
// CHECK:   moore.class.classdecl @C1 implements [@IU] : {
// CHECK:   }
moore.class.classdecl @IU : {
}
moore.class.classdecl @C1 implements [@IU] : {
}

// CHECK-LABEL:   moore.class.classdecl @I1 : {
// CHECK:   }
// CHECK:   moore.class.classdecl @I2 : {
// CHECK:   }
// CHECK:   moore.class.classdecl @C2 implements [@I1, @I2] : {
// CHECK:   }
moore.class.classdecl @I1 : {
}
moore.class.classdecl @I2 : {
}
moore.class.classdecl @C2 implements [@I1, @I2] : {
}

// CHECK-LABEL:   moore.class.classdecl @B : {
// CHECK:   }
// CHECK:   moore.class.classdecl @J1 : {
// CHECK:   }
// CHECK:   moore.class.classdecl @J2 : {
// CHECK:   }
// CHECK:   moore.class.classdecl @D extends @B implements [@J1, @J2] : {
// CHECK:   }
moore.class.classdecl @B : {
}
moore.class.classdecl @J1 : {
}
moore.class.classdecl @J2 : {
}
moore.class.classdecl @D extends @B implements [@J1, @J2] : {
}

// CHECK-LABEL:   moore.class.classdecl @PropertyCombo : {
// CHECK-NEXT:     moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:     moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:     moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK:   }
moore.class.classdecl @PropertyCombo : {
  moore.class.propertydecl @pubAutoI32 : !moore.i32
  moore.class.propertydecl @protStatL18 : !moore.l18
  moore.class.propertydecl @localAutoI32 : !moore.i32
}

// CHECK-LABEL:   moore.module @testModule6() {
// CHECK:    %t = moore.variable : <class.object<@"testModule6::testModuleClass2">>
// CHECK:     %result = moore.variable : <i32>
// CHECK:     moore.procedure initial {
// CHECK:       %0 = moore.class.new : <@"testModule6::testModuleClass2">
// CHECK:       moore.blocking_assign %t, %0 : class.object<@"testModule6::testModuleClass2">
// CHECK:       %1 = moore.class.upcast %t : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
// CHECK:       %2 = moore.class.property_ref %1[@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
// CHECK:       %3 = moore.read %2 : <i32>
// CHECK:       moore.blocking_assign %result, %3 : i32
// CHECK:       %4 = moore.class.upcast %t : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
// CHECK:       %5 = moore.class.call @"testModule6::testModuleClass::returnA"(%4) : (!moore.ref<class.object<@"testModule6::testModuleClass">>) -> (!moore.i32)
// CHECK:       moore.blocking_assign %result, %5 : i32
// CHECK:       moore.return
// CHECK:     }
// CHECK:     moore.output
// CHECK:   }
// CHECK:   moore.class.classdecl @"testModule6::testModuleClass" : {
// CHECK:     moore.class.propertydecl @a : !moore.i32
// CHECK:     moore.class.methoddecl @returnA : (!moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32
// CHECK:  } attributes {sym_visibility = "private"}
// CHECK:   func.func private @"testModule6::testModuleClass::returnA"(%arg0: !moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32 {
// CHECK:     %0 = moore.class.property_ref %arg0[@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
// CHECK:     %1 = moore.read %0 : <i32>
// CHECK:     return %1 : !moore.i32
// CHECK:   }
// CHECK:   moore.class.classdecl @"testModule6::testModuleClass2" extends @"testModule6::testModuleClass" : {
// CHECK:   } attributes {sym_visibility = "private"}

  moore.module @testModule6() {
    %t = moore.variable : <class.object<@"testModule6::testModuleClass2">>
    %result = moore.variable : <i32>
    moore.procedure initial {
      %0 = moore.class.new : <@"testModule6::testModuleClass2">
      moore.blocking_assign %t, %0 : class.object<@"testModule6::testModuleClass2">
      %1 = moore.class.upcast %t : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
      %2 = moore.class.property_ref %1[@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
      %3 = moore.read %2 : <i32>
      moore.blocking_assign %result, %3 : i32
      %4 = moore.class.upcast %t : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
      %5 = moore.class.call @"testModule6::testModuleClass::returnA"(%4) : (!moore.ref<class.object<@"testModule6::testModuleClass">>) -> (!moore.i32)
      moore.blocking_assign %result, %5 : i32
      moore.return
    }
    moore.output
  }
  moore.class.classdecl @"testModule6::testModuleClass" : {
    moore.class.propertydecl @a : !moore.i32
    moore.class.methoddecl @returnA : (!moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32
  } attributes {sym_visibility = "private"}
  func.func private @"testModule6::testModuleClass::returnA"(%arg0: !moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32 {
    %0 = moore.class.property_ref %arg0[@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
    %1 = moore.read %0 : <i32>
    return %1 : !moore.i32
  }
  moore.class.classdecl @"testModule6::testModuleClass2" extends @"testModule6::testModuleClass" : {
  } attributes {sym_visibility = "private"}
}
