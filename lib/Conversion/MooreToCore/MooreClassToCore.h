//===- ClassLoweringUtils.h - Moore class → LLVM helpers -------*- C++ -*-===//
//
// Small helpers to lower moore.class.classdecl into identified LLVM structs.
// - Stable name mangling
// - Opaque struct creation/lookup
// - Body construction (base prefix + properties)
// - Optional cache
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_MOORETOCORE_MOORECLASSTOCORE_H
#define CIRCT_CONVERSION_MOORETOCORE_MOORECLASSTOCORE_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir {
class MLIRContext;
class Type;
class TypeConverter;
} // namespace mlir

namespace mlir::LLVM {
class LLVMStructType;
}

namespace circt::moore {
class ClassDeclOp;
}

namespace circt::moore::llclass {

/// Cache for identified structs keyed by class symbol.
/// Cache for identified structs and field GEP paths keyed by class symbol.
struct ClassTypeCache {
  struct Info {
    mlir::LLVM::LLVMStructType ident;
    // field name -> GEP path inside ident (excluding the leading pointer index)
    llvm::DenseMap<llvm::StringRef, llvm::SmallVector<unsigned, 2>> fieldPath;
  };

  // Keyed by the SymbolRefAttr of the class (use the *same* attr you lower
  // from).
  llvm::DenseMap<mlir::Attribute, Info> map;

  /// Record the identified struct for a class.
  void setIdent(mlir::SymbolRefAttr classSym,
                mlir::LLVM::LLVMStructType ident) {
    map[classSym].ident = ident;
  }

  /// Record/overwrite all field paths for this class.
  void setFieldPath(mlir::SymbolRefAttr classSym, llvm::StringRef fieldName,
                    llvm::ArrayRef<unsigned> path) {
    auto &info = map[classSym];
    info.fieldPath[fieldName] =
        llvm::SmallVector<unsigned, 2>(path.begin(), path.end());
  }

  /// Lookup the identified struct for a class.
  std::optional<mlir::LLVM::LLVMStructType>
  getIdent(mlir::SymbolRefAttr classSym) const {
    if (auto it = map.find(classSym); it != map.end())
      return it->second.ident;
    return std::nullopt;
  }

  /// Lookup the full GEP path for a (class, field).
  std::optional<llvm::ArrayRef<unsigned>>
  getFieldPath(mlir::SymbolRefAttr classSym,
               mlir::FlatSymbolRefAttr fieldSym) const {
    if (auto it = map.find(classSym); it != map.end()) {
      if (auto jt = it->second.fieldPath.find(fieldSym.getValue());
          jt != it->second.fieldPath.end())
        return llvm::ArrayRef<unsigned>(jt->second);
    }
    return std::nullopt;
  }

  /// Legacy helper: only returns a single index if no base is present.
  std::optional<unsigned>
  getFieldIndexFor(mlir::SymbolRefAttr classSym,
                   mlir::FlatSymbolRefAttr fieldSym) const {
    if (auto p = getFieldPath(classSym, fieldSym)) {
      if (p->size() == 1)
        return (*p)[0];
    }
    return std::nullopt; // derived or inherited needs full path
  }
};

mlir::LLVM::LLVMFuncOp getOrCreateMalloc(mlir::ModuleOp module, mlir::OpBuilder &b);

/// Mangle a stable, LLVM-safe identified name from a class symbol.
/// e.g. @"pkg::Outer::C" → "moore.class.pkg.Outer.C"
std::string mangleClassName(mlir::SymbolRefAttr sym);

/// Lookup (or create) an opaque identified struct for this class.
/// Does not set a body.
mlir::LLVM::LLVMStructType getOrCreateOpaqueStruct(mlir::MLIRContext *ctx,
                                                   mlir::SymbolRefAttr sym);

/// Compute and set the body of the identified struct:
/// - If op has a base class, insert the base’s identified struct as the first
/// element.
/// - Then append each property’s lowered LLVM type in declaration order.
/// Returns failure if any child type can’t be converted or if setting the body
/// fails.
mlir::LogicalResult setClassStructBody(circt::moore::ClassDeclOp op,
                                       mlir::LLVM::LLVMStructType ident,
                                       mlir::TypeConverter const &typeConverter,
                                       ClassTypeCache& cache);

} // namespace circt::moore::llclass

#endif // CIRCT_CONVERSION_MOORETOCORE_MOORECLASSTOCORE_H
