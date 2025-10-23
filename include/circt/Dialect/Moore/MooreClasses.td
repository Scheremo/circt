#ifndef CIRCT_DIALECT_MOORE_MOORECLASSES
#define CIRCT_DIALECT_MOORE_MOORECLASSES

include "circt/Dialect/Moore/MooreAttributes.td"
include "circt/Dialect/Moore/MooreDialect.td"
include "circt/Dialect/Moore/MooreTypes.td"
include "circt/Dialect/Moore/MooreOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Base class for the operations in this dialect.
class MooreClassOp<string mnemonic, list<Trait> traits = []> :
  Op<MooreDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Core Class Ops
//===----------------------------------------------------------------------===//

// moore.class.new : @C args
// Allocates and constructs a new object. The constructor is implied by @C$ctor.
// Lowering chooses heap allocation and writes the vptr. Side-effecting.

def ClassNewOp : MooreClassOp<"class.new"> {
  let summary = "Allocate and construct a new class instance";
  let description = [{
    Allocates a new instance of class @C and invokes its constructor with the
    given arguments. The result is a non-null `!moore.class<@C>` handle with
    refcount initialized to 1 (see retain/release ops).
  }];

  let arguments = (ins);
  let results   = (outs ClassHandleType:$result);

  let assemblyFormat = [{
     `:` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

// moore.class.delete %inst : !moore.class.object<@C>
// Calls the destructor (if present) and frees memory. Internal-only in SV.

def ClassDeleteOp : MooreClassOp<"class.delete"> {
  let summary = "Destruct and deallocate a class instance";
  let arguments = (ins ClassHandleType:$instance);
  let assemblyFormat = "$instance `:` type($instance) attr-dict";
}

// moore.class.property_ref %inst[@property] : <!moore.class<@C>> -> <T>
// Produces a reference to a property inside the object. No side effects.
// NOTE: The result is expected to be a `!moore.ref<T>`; we leave the type
// constraint generic here and enforce it in the C++ verifier.

def ClassPropertyRefOp : MooreClassOp<"class.property_ref", [Pure]> {
  let summary = "Get a !moore.ref to a class property";
  let arguments = (ins ClassHandleRefType:$instance, FlatSymbolRefAttr:$property);
  let results   = (outs RefType:$propertyRef);
  let assemblyFormat = "$instance`[`$property`]` `:` type($instance) `->` type($propertyRef) attr-dict";
  let hasVerifier = 1;
}

// moore.class.call @C::m(%inst, args) : (types) -> (results)
// Direct, non-virtual call. Uses CallOpInterface.

def ClassCallOp : MooreOp<"class.call", [CallOpInterface, ArgAndResultAttrsOpInterface]> {
  let summary = "Direct call to a (non-virtual) class method";
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results   = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $operands `)` `:` `(`type($operands)`)` `->` `(` type($results) `)` attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getOperands();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return CallInterfaceCallable(getCalleeAttr());
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
         if (auto sym = callee.dyn_cast<mlir::SymbolRefAttr>())
            setCalleeAttr(sym);
         else
             llvm_unreachable("Expected SymbolRefAttr for callee");
    }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

def ClassVCallOp : MooreOp<"class.vcall", [CallOpInterface, ArgAndResultAttrsOpInterface]> {
  let summary = "Virtual dispatch to a class method";
  let description = [{
    Performs dynamic dispatch by loading the vptr from `%inst` and calling the
    implementation in the vtable slot corresponding to `@Base::m`.
  }];
  let arguments = (ins SymbolRefAttr:$baseMethod, Variadic<AnyType>:$operands);
  let results   = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$baseMethod `(` $operands `)` `:` `(`type($operands)`)` `->` `(` type($results) `)` attr-dict";
  let hasVerifier = 1;


  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return getOperands();
    }
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    mlir::CallInterfaceCallable getCallableForCallee() {
      return CallInterfaceCallable(getBaseMethodAttr());
    }

    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
         if (auto sym = callee.dyn_cast<mlir::SymbolRefAttr>())
            setBaseMethodAttr(sym);
         else
             llvm_unreachable("Expected SymbolRefAttr for callee");
    }

    /// Stub implementations for ArgumentAttributesMethods. If desired,
    /// implement these by defining arg_attrs and res_attrs as arguments to the
    /// operation as OptionalAttr<DictArrayAttr>.
    mlir::ArrayAttr getArgAttrsAttr() { return nullptr; }
    mlir::ArrayAttr getResAttrsAttr() { return nullptr; }
    void setArgAttrsAttr(mlir::ArrayAttr args) {}
    void setResAttrsAttr(mlir::ArrayAttr args) {}
    mlir::Attribute removeArgAttrsAttr() { return nullptr; }
    mlir::Attribute removeResAttrsAttr() { return nullptr; }
  }];
}

// moore.class.upcast %inst : !moore.class<@D> to !moore.class<@B>

def ClassUpcastOp : MooreClassOp<"class.upcast", [Pure]> {
  let summary = "Upcast a derived handle to a base handle (zero-cost)";
  let arguments = (ins ClassHandleRefType:$instance);
  let results   = (outs ClassHandleRefType:$result);
  let assemblyFormat = "$instance `:` type($instance) `to` type($result) attr-dict";
  let hasVerifier = 1;
}

// moore.class.downcast %inst : !moore.class<@B> to !moore.class<@D>
// Returns null on failure (SV $cast). Runtime checks are verified in C++.

def ClassDowncastOp : MooreClassOp<"class.downcast"> {
  let summary = "Downcast a base handle to a derived handle (dynamic check)";
  let arguments = (ins ClassHandleRefType:$instance);
  let results   = (outs ClassHandleRefType:$result);
  let assemblyFormat = "$instance `:` type($instance) `to` type($result) attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Class Declarations (metadata), VTable, and RTTI materialization ops
// These are module-level ops that hold metadata for verification and codegen.
//===----------------------------------------------------------------------===//

def ClassPropertyDeclOp : MooreClassOp < "class.propertydecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class property (variable)";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type
  );

  let results = (outs);
  // Printed as:  moore.class.propertydecl @foo : !moore.ref<!moore.i32> { ... }
  let assemblyFormat = [{
     $sym_name `:` $type  attr-dict
  }];

  let extraClassDeclaration = [{
    // Convenience typed accessor for the stored type.
    ::mlir::Type getPropertyType() { return getTypeAttr().getValue(); }
  }];
}

def ClassMethodDeclOp : MooreClassOp < "class.methoddecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class method";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );

  let results = (outs);
  let assemblyFormat = [{
      $sym_name `:` $function_type  attr-dict
  }];
}

def ClassBodyEndOp : MooreClassOp<"class.declend", [Terminator, HasParent<"ClassDeclOp">]> {
  let summary = "end of classdecl region";
  let assemblyFormat = "attr-dict";
}

// moore.class.classdecl @C { base = @B?, flags = #moore.classflags<...>,
//                      ... regions/attrs ... }
// Serves as the anchor symbol for !moore.class<@C> parameters.

def ClassDeclOp : MooreClassOp < "class.classdecl",
    [Symbol, SymbolTable, IsolatedFromAbove,
     SingleBlockImplicitTerminator<"circt::moore::ClassBodyEndOp">]> {
  let summary = "Class declaration";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<SymbolRefAttr>:$base,
    OptionalAttr<SymbolRefArrayAttr>:$implementedInterfaces
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $sym_name
    (`extends` $base^)?
    (`implements` $implementedInterfaces^)?
    `:` $body attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
