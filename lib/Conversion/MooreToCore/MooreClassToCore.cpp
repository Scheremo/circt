//===- ClassLoweringUtils.cpp - Moore class → LLVM helpers ---------------===//

#include "./MooreClassToCore.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace circt::moore;

namespace circt::moore::llclass {

using LLVMStructType = LLVM::LLVMStructType;

std::string mangleClassName(SymbolRefAttr sym) {
  std::string s;
  llvm::raw_string_ostream os(s);
  os << "moore.class.";
  StringRef root = sym.getRootReference();
  SmallVector<StringRef> parts;
  root.split(parts, "::");
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i)
      os << '.';
    os << parts[i];
  }
  return os.str();
}

LLVMStructType getOrCreateOpaqueStruct(MLIRContext *ctx, SymbolRefAttr sym) {
  auto name = mangleClassName(sym);
  if (auto existing = LLVMStructType::getIdentified(ctx, name))
    return existing;
  return LLVMStructType::getIdentified(ctx, name);
}

/// Ensure we have `declare i8* @malloc(i64)` (opaque ptr prints as !llvm.ptr).
LLVM::LLVMFuncOp getOrCreateMalloc(ModuleOp module, OpBuilder &b) {
  if (auto f = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc"))
    return f;

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPointToStart(module.getBody());

  auto i64Ty = IntegerType::get(module.getContext(), 64);
  auto ptrTy = mlir::LLVM::LLVMPointerType::get(module.getContext()); // opaque pointer
  auto fnTy  = mlir::LLVM::LLVMFunctionType::get(ptrTy, {i64Ty}, /*isVarArg=*/false);

  auto fn = b.create<LLVM::LLVMFuncOp>(module.getLoc(), "malloc", fnTy);
  fn.setLinkage(LLVM::Linkage::External);
  return fn;
}

LogicalResult setClassStructBody(circt::moore::ClassDeclOp op,
                                 mlir::LLVM::LLVMStructType ident,
                                 mlir::TypeConverter const &typeConverter,
                                 ClassTypeCache &cache) {
  SmallVector<mlir::Type> elements;

  auto classSym = mlir::SymbolRefAttr::get(op.getSymNameAttr());
  cache.setIdent(classSym, ident);

  // Base-first (prefix) layout for single inheritance.
  unsigned derivedStartIdx = 0;
  if (auto base = op.getBaseAttr()) {
    auto baseIdent = mlir::LLVM::LLVMStructType::getIdentified(
        op.getContext(), mangleClassName(base));
    if (!baseIdent)
      return op.emitOpError() << "base struct for " << base
                              << " not found (expected identified)";
    elements.push_back(baseIdent);
    derivedStartIdx = 1;

    // Inherit base field paths with a leading 0.
    if (auto baseInfo = cache.getIdent(base)) {
      // We need the base's stored paths.
      auto it = cache.map.find(base);
      if (it != cache.map.end()) {
        for (auto &kv : it->second.fieldPath) {
          llvm::SmallVector<unsigned, 2> path;
          path.push_back(0); // into base subobject
          path.append(kv.second.begin(), kv.second.end());
          cache.setFieldPath(classSym, kv.first, path);
        }
      }
    }
  }

  // Properties in source order.
  unsigned i = 0;
  auto &block = op.getBody().front();
  for (mlir::Operation &child : block) {
    if (auto prop = llvm::dyn_cast<circt::moore::ClassPropertyDeclOp>(child)) {
      mlir::Type mooreTy = prop.getPropertyType();
      mlir::Type llvmTy = typeConverter.convertType(mooreTy);
      if (!llvmTy)
        return prop.emitOpError()
               << "failed to convert property type " << mooreTy;

      elements.push_back(llvmTy);

      // Derived field path: either {i} or {1+i} if base is present.
      llvm::SmallVector<unsigned, 2> path{derivedStartIdx + i};
      cache.setFieldPath(classSym, prop.getSymName(), path);
      ++i;
    }
  }

  if (failed(ident.setBody(elements, /*isPacked=*/false)))
    return op.emitOpError() << "failed to set LLVM struct body";
  return mlir::success();
}

} // namespace circt::moore::llclass
