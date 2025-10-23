//===- MooreClasses.cpp - Implement the Moore classes
//-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect class system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace circt::moore;

/// Resolve a symbol relative to the nearest table.
static Operation *lookupNearest(Operation *from, SymbolRefAttr sym) {
  return SymbolTable::lookupNearestSymbolFrom(from, sym);
}

LogicalResult ClassDowncastOp::verify() {
  return emitOpError() << "Downcasts are not supported yet!";
}

LogicalResult ClassUpcastOp::verify() {
  // Types must be !moore.ref<class.object<…>> on both operand and result.
  auto srcRefTy = dyn_cast<RefType>(getOperand().getType());
  if (!srcRefTy)
    return emitOpError("operand must be a !moore.ref<...>");

  auto dstRefTy = dyn_cast<RefType>(getResult().getType());
  if (!dstRefTy)
    return emitOpError("result must be a !moore.ref<...>");

  auto srcHandleTy = dyn_cast<ClassHandleType>(srcRefTy.getNestedType());
  if (!srcHandleTy)
    return emitOpError("operand must be !moore.ref<class.object<...>>; got ")
           << getOperand().getType();

  auto dstHandleTy = dyn_cast<ClassHandleType>(dstRefTy.getNestedType());
  if (!dstHandleTy)
    return emitOpError("result must be !moore.ref<class.object<...>>; got ")
           << getResult().getType();

  // Upcast is only valid if the source class is the same as, or derived from,
  // the destination class (i.e. D -> B where D derives from B).
  mlir::SymbolRefAttr dstSym = dstHandleTy.getClassSym();
  if (!dstSym)
    return emitOpError("result handle type is missing a class symbol");

  if (!srcHandleTy.isSameOrDerivedFrom(getOperation(), dstSym)) {
    return emitOpError("cannot upcast receiver class ")
           << srcHandleTy.getClassSym() << " to base class " << dstSym
           << " (not the same class and not in its base chain)";
  }

  return success();
}

LogicalResult ClassCallOp::verify() {
  // Resolve callee -> func.func
  auto calleeSym = getCalleeAttr();
  if (!calleeSym)
    return emitOpError("missing `callee`");

  Operation *opSym = lookupNearest(getOperation(), calleeSym);
  if (!opSym)
    return emitOpError("callee symbol `") << calleeSym << "` was not found";

  auto fn = dyn_cast<mlir::FunctionOpInterface>(opSym);
  if (!fn)
    return emitOpError("callee `")
           << calleeSym << "` is not a function-like op";

  auto fnType = dyn_cast<mlir::FunctionType>(fn.getFunctionType());
  if (!fnType)
    return emitOpError("callee `")
           << calleeSym << "` does not have FunctionType";

  // Check arity
  if (fnType.getNumInputs() != static_cast<unsigned int>(getOperands().size()))
    return emitOpError("expected ")
           << fnType.getNumInputs() << " operands, got "
           << getOperands().size();

  // Check each operand type matches the function input type (1:1)
  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(getOperands(), fnType.getInputs()))) {
    auto [arg, expectedTy] = pair;
    if (arg.getType() != expectedTy)
      return emitOpError("operand #")
             << i << " type " << arg.getType()
             << " does not match callee input type " << expectedTy;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ClassVCallOp verifier
// Virtual dispatch to a method declared in a class.
// baseMethod must be a nested symbol: @Class::@method
//===----------------------------------------------------------------------===//

LogicalResult ClassVCallOp::verify() {
  // baseMethod must be present and of the shape @Class::@method (i.e. nested).
  SymbolRefAttr baseSym = getBaseMethodAttr();
  if (!baseSym)
    return emitOpError("missing `baseMethod`");

  auto nested = baseSym.getNestedReferences();
  if (nested.empty())
    return emitOpError(
        "`baseMethod` must be a nested symbol of the form @Class::@method");

  // Resolve the class declaration from the root part.
  // getRootReference() is a StringAttr; wrap it for lookup.
  SymbolRefAttr rootRefForLookup =
      SymbolRefAttr::get(baseSym.getRootReference());
  Operation *classOp = lookupNearest(getOperation(), rootRefForLookup);
  auto classDecl = dyn_cast_or_null<ClassDeclOp>(classOp);
  if (!classDecl)
    return emitOpError("root of `baseMethod` (")
           << baseSym.getRootReference()
           << ") is not a `moore.class.classdecl`";

  // Resolve the method declaration within the class’ symbol table.
  FlatSymbolRefAttr methodLeaf = nested.back();
  Operation *methodOp =
      SymbolTable::lookupSymbolIn(classDecl, methodLeaf.getAttr());
  auto methodDecl = dyn_cast_or_null<ClassMethodDeclOp>(methodOp);
  if (!methodDecl)
    return emitOpError("`")
           << methodLeaf << "` is not a `moore.class.methoddecl` in class "
           << baseSym.getRootReference();

  // Receiver checks: first operand must be !moore.ref<class.object<...>>.
  if (getNumOperands() == 0)
    return emitOpError("virtual call requires a receiver operand");

  auto recvRefTy = dyn_cast<RefType>(getOperand(0).getType());
  if (!recvRefTy)
    return emitOpError("first operand (receiver) must be a !moore.ref<...>");

  auto recvHandleTy = dyn_cast<ClassHandleType>(recvRefTy.getNestedType());
  if (!recvHandleTy)
    return emitOpError("first operand (receiver) must be "
                       "!moore.ref<class.object<...>>; got ")
           << getOperand(0).getType();

  // Subtype check: receiver class derives from the base class in `baseMethod`.
  SymbolRefAttr recvClassSym = recvHandleTy.getClassSym();
  // NOTE: helper takes StringAttr as the base (root) name:
  if (!recvHandleTy.isSameOrDerivedFrom(getOperation(), baseSym))
    return emitOpError("receiver class ")
           << recvClassSym
           << " is not the same as, or derived from, base class "
           << baseSym.getRootReference();

  // Signature check against the declared method function type.
  auto methodFTy =
      llvm::cast<FunctionType>(methodDecl.getFunctionTypeAttr().getValue());

  // Compare operand types to inputs (method type includes %this as first
  // input).
  auto inputs = methodFTy.getInputs();
  auto outputs = methodFTy.getResults();

  if (getOperands().size() != inputs.size())
    return emitOpError("operand count (")
           << getOperands().size()
           << ") does not match method parameter count (" << inputs.size()
           << ") for " << baseSym;

  for (auto it : llvm::enumerate(getOperands())) {
    Type got = it.value().getType();
    Type exp = inputs[it.index()];
    if (got != exp)
      return emitOpError("operand #") << it.index() << " type mismatch: got "
                                      << got << ", expected " << exp;
  }

  if (getResultTypes().size() != outputs.size())
    return emitOpError("result count (")
           << getResultTypes().size()
           << ") does not match method result count (" << outputs.size()
           << ") for " << baseSym;

  for (auto it : llvm::enumerate(getResultTypes())) {
    Type got = it.value();
    Type exp = outputs[it.index()];
    if (got != exp)
      return emitOpError("result #") << it.index() << " type mismatch: got "
                                     << got << ", expected " << exp;
  }

  return success();
}

LogicalResult ClassPropertyRefOp::verify() {
  // The operand is constrained to ClassHandleRefType in ODS; unwrap it.
  auto instRefTy = cast<RefType>(getInstance().getType());
  if (!instRefTy)
    return emitOpError("instance is not a !moore.ref<...>");

  auto handleTy = dyn_cast<ClassHandleType>(instRefTy.getNestedType());
  if (!handleTy)
    return emitOpError("instance must be !moore.ref<class.object<@C>>");

  // Extract the referenced class symbol from the handle type.
  SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("instance type is missing a class symbol");

  // Resolve the class symbol starting from the nearest symbol table.
  Operation *clsSym = lookupNearest(getOperation(), classSym);
  if (!clsSym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";
  auto classDecl = dyn_cast<ClassDeclOp>(clsSym);
  if (!classDecl)
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  // Look up the field symbol inside the class declaration's symbol table.
  FlatSymbolRefAttr fieldSym = getPropertyAttr();
  if (!fieldSym)
    return emitOpError("missing field symbol");

  Operation *fldSym =
      SymbolTable::lookupSymbolIn(classDecl, fieldSym.getAttr());
  if (!fldSym)
    return emitOpError("no field `") << fieldSym << "` in class " << classSym;

  auto fieldDecl = dyn_cast<ClassPropertyDeclOp>(fldSym);
  if (!fieldDecl)
    return emitOpError("symbol `")
           << fieldSym << "` is not a `moore.class.propertydecl`";

  // Result must be !moore.ref<T> where T matches the field's declared type.
  auto resRefTy = cast<RefType>(getPropertyRef().getType());
  if (!resRefTy)
    return emitOpError("result must be a !moore.ref<T>");

  Type expectedElemTy = fieldDecl.getPropertyType();
  if (resRefTy.getNestedType() != expectedElemTy)
    return emitOpError("result element type (")
           << resRefTy.getNestedType() << ") does not match field type ("
           << expectedElemTy << ")";

  return success();
}

LogicalResult ClassNewOp::verify() {
  // The result is constrained to ClassHandleType in ODS, so this cast should be
  // safe.
  auto handleTy = cast<ClassHandleType>(getResult().getType());
  if (!handleTy)
    return emitOpError("result type not a ClassHandleType");

  mlir::SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("result type is missing a class symbol");

  // Resolve the referenced symbol starting from the nearest symbol table.
  mlir::Operation *sym = lookupNearest(getOperation(), classSym);
  if (!sym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";

  if (!llvm::isa<ClassDeclOp>(sym))
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  return mlir::success();
}

LogicalResult ClassDeclOp::verify() {
  auto &block = getBody().front();
  for (mlir::Operation &op : block) {

    // allow only property and method decls and terminator
    if (llvm::isa<circt::moore::ClassBodyEndOp, circt::moore::ClassPropertyDeclOp,
                  circt::moore::ClassMethodDeclOp>(&op))
      continue;

    return emitOpError() << "body may only contain 'moore.class.fielddecl' and "
                            "'moore.class.methoddecl' operations";
  }
  return mlir::success();
}
