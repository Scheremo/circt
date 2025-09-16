//===- MooreAttributes.td - Moore attribute definitions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
#define CIRCT_DIALECT_MOORE_MOOREATTRIBUTES

include "circt/Dialect/Moore/MooreDialect.td"
include "mlir/IR/AttrTypeBase.td"

def FVIntegerAttr : AttrDef<MooreDialect, "FVInteger"> {
  let mnemonic = "fvint";
  let summary = "An attribute containing a four-valued integer";
  let parameters = (ins "FVInt":$value);
  let hasCustomAssemblyFormat = 1;
  let convertFromStorage = "$_self.getValue()";
  let returnType = "circt::FVInt";
}

def TimeFormatAttr : AttrDef<MooreDialect, "TimeFormat"> {
  let mnemonic = "timeformat";
  let summary = "An attribute holding timeformat state";
  let description = [{
    Encodes the four fields controlled by SystemVerilog `$timeformat`:
    units exponent, precision exponent, string suffix, and minimum field width.
  }];
  let parameters = (ins
    "int8_t":$units_number,
    "int8_t":$precision_number,
    "mlir::StringAttr":$suffix_string,
    "uint8_t":$minimum_field_width
  );
  let hasCustomAssemblyFormat = 1;
}

#endif // CIRCT_DIALECT_MOORE_MOOREATTRIBUTES
