// RUN: circt-verilog %s --parse-only | FileCheck %s

/// Flag tests

// CHECK-LABEL: moore.class.classdecl @plain : {
// CHECK: }
class plain;
endclass

// CHECK-LABEL: moore.class.classdecl @abstractOnly : {
// CHECK: }
virtual class abstractOnly;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass : {
// CHECK: }
interface class interfaceTestClass;
endclass

/// Interface tests

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass2 implements [@interfaceTestClass] : {
// CHECK: }
class interfaceTestClass2 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass3 implements [@interfaceTestClass] : {
// CHECK: }
interface class interfaceTestClass3 extends interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass4 implements [@interfaceTestClass3] : {
// CHECK: }
class interfaceTestClass4 implements interfaceTestClass3;
endclass

/// Inheritance tests

// CHECK-LABEL: moore.class.classdecl @inheritanceTest : {
// CHECK: }
class inheritanceTest;
endclass

// CHECK-LABEL: moore.class.classdecl @inheritanceTest2 extends @inheritanceTest : {
// CHECK: }
class inheritanceTest2 extends inheritanceTest;
endclass

// Inheritance + interface tests

// CHECK-LABEL: moore.class.classdecl @D extends @plain : {
// CHECK: }
class D extends plain;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl1 implements [@interfaceTestClass] : {
// CHECK: }
class Impl1 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl2 implements [@interfaceTestClass, @interfaceTestClass3] : {
// CHECK: }
class Impl2 implements interfaceTestClass, interfaceTestClass3;
endclass

// CHECK-LABEL: moore.class.classdecl @DI extends @D implements [@interfaceTestClass] : {
// CHECK: }
class DI extends D implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @IMulti implements [@interfaceTestClass, @interfaceTestClass3] : {
// CHECK: }
interface class IMulti extends interfaceTestClass, interfaceTestClass3;
endclass

/// Property tests

// CHECK-LABEL: moore.class.classdecl @PropertyCombo : {
// CHECK:   moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:   moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK: }
class PropertyCombo;
  // public automatic int
  int pubAutoI32;

  // protected static logic [17:0]
  protected static logic [17:0] protStatL18;

  // local automatic int
  local int localAutoI32;
endclass

// Ensure multiple propertys preserve declaration order
// CHECK-LABEL: moore.class.classdecl @PropertyOrder : {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @b : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @c : !moore.i32
// CHECK: }
class PropertyOrder;
  int a;
  int b;
  int c;
endclass

// Classes within packages
package testPackage;
   // CHECK-LABEL: moore.class.classdecl @"testPackage::testPackageClass" : {
   class testPackageClass;
   // CHECK: }
   endclass
endpackage

// CHECK-LABEL: moore.module @testModule() {
// CHECK: }
// CHECK: moore.class.classdecl @"testModule::testModuleClass" : {
// CHECK: }
module testModule #();
   class testModuleClass;
   endclass
endmodule

// CHECK-LABEL: moore.class.classdecl @testClass : {
// CHECK: }
// CHECK: moore.class.classdecl @"testClass::testClass" : {
// CHECK: }
class testClass;
   class testClass;
   endclass // testClass
endclass

/// Check handle declaration

// CHECK-LABEL:  moore.module @testModule2() {
module testModule2 #();
    class testModuleClass;
    endclass // testModuleClass2
    // CHECK-NEXT: [[OBJ:%.+]] = moore.variable : <class.object<@"testModule2::testModuleClass">>
    testModuleClass t;
    // CHECK-NEXT:     moore.output
    // CHECK-NEXT:   }
endmodule
// CHECK: moore.class.classdecl @"testModule2::testModuleClass" : {
// CHECK: }

/// Check calls to new

// CHECK-LABEL: moore.module @testModule3() {
module testModule3;
    class testModuleClass;
    endclass
    // CHECK: [[T:%.*]] = moore.variable : <class.object<@"testModule3::testModuleClass">>
    testModuleClass t;
    // CHECK: moore.procedure initial {
    initial begin
        // CHECK:   [[NEW:%.*]] = moore.class.new : <@"testModule3::testModuleClass">
        // CHECK:   moore.blocking_assign [[T]], [[NEW]] : class.object<@"testModule3::testModuleClass">
        t = new;
        // CHECK:   moore.return
        // CHECK: }
    end
    // CHECK: moore.output
endmodule

/// Check concrete method calls

// CHECK-LABEL: moore.module @testModule4() {
// CHECK: [[T:%.*]] = moore.variable : <class.object<@"testModule4::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule4::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class.object<@"testModule4::testModuleClass">
// CHECK:    [[FUNCRET:%.+]] = moore.class.call @"testModule4::testModuleClass::returnA"([[T]]) : (!moore.ref<class.object<@"testModule4::testModuleClass">>) -> (!moore.i32)
// CHECK:    moore.blocking_assign [[RESULT]], [[FUNCRET]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule4::testModuleClass" : {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK-NEXT: moore.class.methoddecl @returnA : (!moore.ref<class.object<@"testModule4::testModuleClass">>) -> !moore.i32
// CHECK: }

// CHECK: func.func private @"testModule4::testModuleClass::returnA"
// CHECK-SAME: ([[ARG:%.+]]: !moore.ref<class.object<@"testModule4::testModuleClass">>)
// CHECK-SAME: -> !moore.i32 {
// CHECK-NEXT: [[REF:%.+]] = moore.class.property_ref [[ARG]][@a] : <class.object<@"testModule4::testModuleClass">> -> <i32>
// CHECK-NEXT: [[RETURN:%.+]] = moore.read [[REF]] : <i32>
// CHECK-NEXT: return [[RETURN]] : !moore.i32
// CHECK-NEXT: }

module testModule4;
    class testModuleClass;
       int a;
       function int returnA();
          return a;
       endfunction
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end
endmodule

/// Check inherited concrete method calls

// CHECK-LABEL: moore.module @testModule5()
// CHECK: [[T:%.*]] = moore.variable : <class.object<@"testModule5::testModuleClass2">>
// CHECK: [[RESULT:%.*]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK-NEXT:   [[NEW:%.*]] = moore.class.new : <@"testModule5::testModuleClass2">
// CHECK-NEXT:   moore.blocking_assign [[T]], [[NEW]] : class.object<@"testModule5::testModuleClass2">
// CHECK-NEXT:   [[UP:%.*]] = moore.class.upcast [[T]] : <class.object<@"testModule5::testModuleClass2">> to <class.object<@"testModule5::testModuleClass">>
// CHECK-NEXT:   [[RET:%.*]] = moore.class.call @"testModule5::testModuleClass::returnA"([[UP]]) : (!moore.ref<class.object<@"testModule5::testModuleClass">>) -> (!moore.i32)
// CHECK-NEXT:   moore.blocking_assign [[RESULT]], [[RET]] : i32
// CHECK:   moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule5::testModuleClass" : {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK:   moore.class.methoddecl @returnA : (!moore.ref<class.object<@"testModule5::testModuleClass">>) -> !moore.i32
// CHECK: }

// CHECK: func.func private @"testModule5::testModuleClass::returnA"(%arg0: !moore.ref<class.object<@"testModule5::testModuleClass">>) -> !moore.i32 {
// CHECK:   [[FREF:%.*]] = moore.class.property_ref %arg0[@a] : <class.object<@"testModule5::testModuleClass">> -> <i32>
// CHECK:   [[READ:%.*]] = moore.read [[FREF]] : <i32>
// CHECK:   return [[READ]] : !moore.i32
// CHECK: }

// CHECK: moore.class.classdecl @"testModule5::testModuleClass2" extends @"testModule5::testModuleClass" : {
// CHECK: }

module testModule5;
    class testModuleClass;
       int a;
       function int returnA();
          return a;
       endfunction
    endclass
   class testModuleClass2 extends testModuleClass;
   endclass

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end

endmodule

/// Check inherited property access

 // CHECK-LABEL: moore.module @testModule6() {
 // CHECK:    [[t:%.+]] = moore.variable : <class.object<@"testModule6::testModuleClass2">>
 // CHECK:    [[result:%.+]] = moore.variable : <i32>
 // CHECK:    moore.procedure initial {
 // CHECK:      [[NEW:%.+]] = moore.class.new : <@"testModule6::testModuleClass2">
 // CHECK:      moore.blocking_assign [[t]], [[NEW]] : class.object<@"testModule6::testModuleClass2">
 // CHECK:      [[UPCAST:%.+]] = moore.class.upcast [[t]] : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
 // CHECK:      [[FIELDREF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
 // CHECK:      [[VALUE:%.+]] = moore.read [[FIELDREF]] : <i32>
 // CHECK:      moore.blocking_assign %result, [[VALUE]] : i32
 // CHECK:     [[UPCAST:%.+]] = moore.class.upcast %t : <class.object<@"testModule6::testModuleClass2">> to <class.object<@"testModule6::testModuleClass">>
 // CHECK:     [[FIELDREF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
 // CHECK:     [[CONST:%.+]] = moore.constant 3 : i32
 // CHECK:     moore.blocking_assign [[FIELDREF]], [[CONST]] : i32
 // CHECK:      moore.return
 // CHECK:    }
 // CHECK:    moore.output
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule6::testModuleClass" : {
 // CHECK:    moore.class.propertydecl @a : !moore.i32
 // CHECK:    moore.class.methoddecl @returnA : (!moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32
 // CHECK:  }
 // CHECK:  func.func private @"testModule6::testModuleClass::returnA"(%arg0: !moore.ref<class.object<@"testModule6::testModuleClass">>) -> !moore.i32 {
 // CHECK:    [[FIELDREF:%.+]] = moore.class.property_ref %arg0[@a] : <class.object<@"testModule6::testModuleClass">> -> <i32>
 // CHECK:    [[VALUE:%.+]] = moore.read [[FIELDREF]] : <i32>
 // CHECK:    return [[VALUE]] : !moore.i32
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule6::testModuleClass2" extends @"testModule6::testModuleClass" : {
 // CHECK:  }

module testModule6;

    class testModuleClass;
       int a;
       function int returnA();
          return a;
       endfunction
    endclass // testModuleClass

   class testModuleClass2 extends testModuleClass;
   endclass // testModuleClass2

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.a;
        t.a = 3;
    end

endmodule
