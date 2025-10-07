//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// VerilogTextFile.cpp
//
// This file implements the VerilogTextFile class, a lightweight wrapper around
// VerilogDocument that represents the open text buffer for a single source file
// managed by the CIRCT Verilog LSP server.
//
// Responsibilities:
//   * Manage the current text contents and version of an open Verilog file.
//   * Rebuild the associated VerilogDocument whenever the file is opened or
//     updated via LSP “didOpen” or “didChange” notifications.
//   * Apply incremental text changes as specified by the LSP protocol.
//   * Forward symbol-definition and reference queries to the underlying
//     VerilogDocument.
//
// The class acts as the LSP-facing façade for each active text document,
// maintaining an editable in-memory copy of its contents while keeping the
// corresponding Slang-based VerilogDocument synchronized for semantic analysis.
//
//===----------------------------------------------------------------------===//

#include "VerilogTextFile.h"
#include "../Utils/LSPUtils.h"
#include "VerilogDocument.h"

using namespace circt::lsp;
using namespace llvm;
using namespace llvm::lsp;

VerilogTextFile::VerilogTextFile(
    VerilogServerContext &context, const llvm::lsp::URIForFile &uri,
    StringRef fileContents, int64_t version,
    std::vector<llvm::lsp::Diagnostic> &diagnostics)
    : context(context), contents(fileContents.str()) {
  initialize(uri, version, diagnostics);
}

LogicalResult VerilogTextFile::update(
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    ArrayRef<llvm::lsp::TextDocumentContentChangeEvent> changes,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {

  LogicalResult applied = failure();
  {
    std::unique_lock lk(contentMutex);
    applied =
        llvm::lsp::TextDocumentContentChangeEvent::applyTo(changes, contents);
  }

  if (failed(applied)) {
    circt::lsp::Logger::error(Twine("Failed to update contents of ") +
                              uri.file());
    return failure();
  }

  // If the file contents were properly changed, reinitialize the text file.
  initialize(uri, newVersion, diagnostics);
  return success();
}

void VerilogTextFile::initialize(
    const llvm::lsp::URIForFile &uri, int64_t newVersion,
    std::vector<llvm::lsp::Diagnostic> &diagnostics) {
  std::shared_ptr<VerilogDocument> newDocument;
  std::string newDocContents;

  // Copy contents to avoid blocking change updates
  {
    std::unique_lock lk(contentMutex);
    newDocContents = contents;
  }

  newDocument = std::make_shared<VerilogDocument>(context, uri, newDocContents,
                                                  diagnostics);
  setDocument(newDocument);
  version = newVersion;
}

void VerilogTextFile::getLocationsOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position defPos,
    std::vector<llvm::lsp::Location> &locations) {
  auto doc = getDocument();
  if (doc)
    doc->getLocationsOf(uri, defPos, locations);
}

void VerilogTextFile::findReferencesOf(
    const llvm::lsp::URIForFile &uri, llvm::lsp::Position pos,
    std::vector<llvm::lsp::Location> &references) {
  auto doc = getDocument();
  if (doc)
    doc->findReferencesOf(uri, pos, references);
}

std::shared_ptr<VerilogDocument> VerilogTextFile::getDocument() {
  std::shared_ptr<circt::lsp::VerilogDocument> doc;
  {
    std::unique_lock lk(docMutex);
    doc = document;
  }
  return doc;
}

void VerilogTextFile::setDocument(std::shared_ptr<VerilogDocument> newDoc) {
  std::unique_lock lk(docMutex);
  document = std::move(newDoc);
}
