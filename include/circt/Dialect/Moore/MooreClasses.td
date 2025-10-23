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
// Class Attributes
//===----------------------------------------------------------------------===//

// A bitmask enum whose values can be OR’d together.
def ClassFlags : I32Enum<"ClassFlags", "class flags", [
  I32BitEnumAttrCase<"Concrete",    0x0>,
  I32BitEnumAttrCase<"Abstract",    0x1>,
  I32BitEnumAttrCase<"Interface",   0x2>,
  I32BitEnumAttrCase<"Final",       0x3>
]> {
  let cppNamespace = "circt::moore";
}

// Visibility of a class field (SystemVerilog-ish: public/protected/local)
def FieldVisibility : I32Enum<"FieldVisibility", "class field visibility", [
  I32EnumAttrCase<"Public",    0>,
  I32EnumAttrCase<"Protected", 1>,
  I32EnumAttrCase<"Local",     2>
]> {
  let cppNamespace = "circt::moore";
}

// Lifetime of a class field (automatic or static)
def FieldLifetime : I32Enum<"FieldLifetime", "class field lifetime", [
  I32EnumAttrCase<"Automatic", 0>,
  I32EnumAttrCase<"Static",    1>
]> {
  let cppNamespace = "circt::moore";
}

def ClassFlagsAttr      : EnumAttr<MooreDialect, ClassFlags, "class_flags">;
def FieldVisibilityAttr : EnumAttr<MooreDialect, FieldVisibility, "visibility">;
def FieldLifetimeAttr   : EnumAttr<MooreDialect, FieldLifetime, "lifetime">;

//===----------------------------------------------------------------------===//
// Core Class Ops
//===----------------------------------------------------------------------===//

// moore.class.new @C(args) : (arg_types) -> !moore.class<@C>
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

// moore.class.delete %inst : !moore.class<@C>
// Calls the destructor (if present) and frees memory. Internal-only in SV.

def ClassDeleteOp : MooreClassOp<"class.delete"> {
  let summary = "Destruct and deallocate a class instance";
  let arguments = (ins ClassHandleType:$instance);
  let assemblyFormat = "$instance `:` type($instance) attr-dict";
}

// moore.class.field_ref %inst, @field : -> !moore.ref<T>
// Produces a reference to a field inside the object. No side effects.
// NOTE: The result is expected to be a `!moore.ref<T>`; we leave the type
// constraint generic here and enforce it in the C++ verifier.

def ClassFieldRefOp : MooreClassOp<"class.field_ref", [Pure]> {
  let summary = "Get a !moore.ref to a class field";
  let arguments = (ins ClassHandleRefType:$instance, FlatSymbolRefAttr:$field);
  let results   = (outs RefType:$fieldRef);
  let assemblyFormat = "$instance `:` type($instance) `,` $field `:` type($fieldRef) attr-dict";
  let hasVerifier = 1;
}

// moore.class.call @C::m(%inst, args) : (types) -> (results)
// Direct, non-virtual call. Uses CallOpInterface.

def ClassCallOp : MooreOp<"class.call", [CallOpInterface, ArgAndResultAttrsOpInterface]> {
  let summary = "Direct call to a (non-virtual) class method";
  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results   = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $operands `:` type($operands) `)` (`->` type($results)^)? attr-dict";
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
  let assemblyFormat = "$baseMethod `(` $operands `:` type($operands) `)` (`->` type($results)^)? attr-dict";
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

// moore.class.retain %h / moore.class.release %h
// Reference-count operations inserted by the frontend and RC passes.

//===----------------------------------------------------------------------===//
// Class Object Lifecycle Ops
//===----------------------------------------------------------------------===//

def ClassRetainOp : MooreClassOp<"class.retain"> {
  let summary = "Increment the reference count of a handle";
  let arguments = (ins ClassHandleRefType:$handle);
  let assemblyFormat = "$handle `:` type($handle) attr-dict";
}

def ClassReleaseOp : MooreClassOp<"class.release"> {
  let summary = "Decrement the reference count; delete on zero";
  let arguments = (ins ClassHandleRefType:$handle);
  let assemblyFormat = "$handle `:` type($handle) attr-dict";
}

// moore.class.null : !moore.class<@C>

def ClassNullOp : MooreClassOp<"class.null", [Pure]> {
  let summary = "Create a null class handle constant";
  let results = (outs ClassHandleRefType:$result);
  let assemblyFormat = " `:` type($result) attr-dict";
}

//===----------------------------------------------------------------------===//
// Class RTTI Checks
//===----------------------------------------------------------------------===//

// moore.class.isnull %h : !moore.class<@C> -> i1

def ClassIsNullOp : MooreClassOp<"class.isnull", [Pure]> {
  let summary = "Return 1 if the handle is null";
  let arguments = (ins ClassHandleType:$handle);
  let results   = (outs I1:$isNull);
  let assemblyFormat = "$handle `:` type($handle) attr-dict";
}

// moore.class.isa %h, @Type : i1

def ClassIsaOp : MooreClassOp<"class.isa", [Pure]> {
  let summary = "Dynamic type test against a class symbol";
  let arguments = (ins ClassHandleType:$handle, SymbolRefAttr:$target);
  let results   = (outs I1:$isA);
  let assemblyFormat = "$handle `,` $target `:` type($handle) attr-dict";
}

// moore.class.typename %h : !moore.string (or target string type)
// We leave the result unconstrained to AnyType and verify in C++ to match the
// dialect's canonical string type.

def ClassHandleTypenameOp : MooreClassOp<"class.typename", [Pure]> {
  let summary = "Return the dynamic type name of the handle";
  let arguments = (ins ClassHandleType:$handle);
  let results   = (outs StringType:$name);
  let assemblyFormat = "$handle `:` type($handle) attr-dict";
}

//===----------------------------------------------------------------------===//
// Class Declarations (metadata), VTable, and RTTI materialization ops
// These are module-level ops that hold metadata for verification and codegen.
//===----------------------------------------------------------------------===//

def ClassFieldDeclOp : MooreClassOp < "class.propertydecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class property (variable)";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    FieldVisibilityAttr:$vis,
    FieldLifetimeAttr:$lifetime
  );

  let results = (outs);
  // Printed as:  moore.class.fielddecl @foo : !moore.ref<!moore.i32> { ... }
  let assemblyFormat = [{
      `[` $vis `,` $lifetime `]` $sym_name `:` $type  attr-dict
  }];

  let extraClassDeclaration = [{
    // Convenience typed accessor for the stored type.
    ::mlir::Type getFieldType() { return getTypeAttr().getValue(); }
  }];
}

def ClassMethodDeclOp : MooreClassOp < "class.methoddecl",
  [Symbol, HasParent<"ClassDeclOp">]> {
  let summary = "Declare a class method";

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    FieldVisibilityAttr:$vis,
    FieldLifetimeAttr:$lifetime
  );

  let results = (outs);
  let assemblyFormat = [{
      `[` $vis `,` $lifetime `]` $sym_name `:` $function_type  attr-dict
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
    ClassFlagsAttr:$flag,
    OptionalAttr<SymbolRefAttr>:$base,
    OptionalAttr<SymbolRefArrayAttr>:$implementedInterfaces
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $flag $sym_name
    (`extends` $base^)?
    (`implements` $implementedInterfaces^)?
    `:` $body attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_MOORE_MOORECLASSES
