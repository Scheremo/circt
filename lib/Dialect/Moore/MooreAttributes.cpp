//===- MooreAttributes.cpp - Implement the Moore attributes ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect attributes.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreAttributes.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::moore;
using mlir::AsmParser;
using mlir::AsmPrinter;

//===----------------------------------------------------------------------===//
// FVIntegerAttr
//===----------------------------------------------------------------------===//

Attribute FVIntegerAttr::parse(AsmParser &p, Type) {
  // Parse the value and width specifier.
  FVInt value;
  unsigned width;
  llvm::SMLoc widthLoc;
  if (p.parseLess() || parseFVInt(p, value) || p.parseColon() ||
      p.getCurrentLocation(&widthLoc) || p.parseInteger(width) ||
      p.parseGreater())
    return {};

  // Make sure the integer fits into the requested number of bits.
  unsigned neededBits =
      value.isNegative() ? value.getSignificantBits() : value.getActiveBits();
  if (width < neededBits) {
    p.emitError(widthLoc) << "integer literal requires at least " << neededBits
                          << " bits, but attribute specifies only " << width;
    return {};
  }

  return FVIntegerAttr::get(p.getContext(), value.sextOrTrunc(width));
}

void FVIntegerAttr::print(AsmPrinter &p) const {
  p << "<";
  printFVInt(p, getValue());
  p << " : " << getValue().getBitWidth() << ">";
}

Attribute TimeFormatAttr::parse(mlir::AsmParser &parser, mlir::Type) {
  if (parser.parseLess())
    return {};

  // Parse units_number (int8_t)
  int64_t units64 = 0;
  if (parser.parseInteger(units64))
    return {};
  if (units64 < -15 ||
      units64 > 2)
    return parser.emitError(parser.getNameLoc(),
                            "units_number out of [-15..2] range"),
           Attribute();

  if (parser.parseComma())
    return {};

  // Parse precision_number (int8_t)
  int64_t prec64 = 0;
  if (parser.parseInteger(prec64))
    return {};
  if (prec64 < -15 ||
      prec64 > 2)
    return parser.emitError(parser.getNameLoc(),
                            "precision_number out of [-15..2] range"),
           Attribute();

  if (parser.parseComma())
    return {};

  // suffix_string (StringAttr)
  StringAttr suffixAny;
  if (parser.parseAttribute(suffixAny))
    return {};
  auto suffixStr = ::llvm::dyn_cast<StringAttr>(suffixAny);
  if (!suffixStr) {
    parser.emitError(parser.getNameLoc(), "expected string attribute for suffix_string");
    return {};
  }

  if (parser.parseComma())
    return {};

  // Parse minimum_field_width (uint8_t)
  int64_t width64 = 0;
  if (parser.parseInteger(width64))
    return {};
  if (width64 < std::numeric_limits<uint8_t>::min() ||
      width64 > std::numeric_limits<uint8_t>::max())
    return parser.emitError(parser.getNameLoc(),
                            "minimum_field_width out of uint8_t range"),
           Attribute();

  if (parser.parseGreater())
    return {};

  // Build the attribute.
  auto *ctx = parser.getContext();
  return TimeFormatAttr::get(ctx, static_cast<int8_t>(units64),
                             static_cast<int8_t>(prec64), suffixAny,
                             static_cast<uint8_t>(width64));
}


void TimeFormatAttr::print(mlir::AsmPrinter &p) const {
  p << "<" << int(getUnitsNumber()) << ", " << int(getPrecisionNumber())
    << ", ";
  p.printAttribute(getSuffixString());
  p << ", " << int(getMinimumFieldWidth()) << ">";
}

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Moore/MooreAttributes.cpp.inc"

void MooreDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Moore/MooreAttributes.cpp.inc"
      >();
}
