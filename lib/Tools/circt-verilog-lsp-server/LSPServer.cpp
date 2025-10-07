//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LSPServer.h"
#include "Utils/DocChangeBucket.h"
#include "VerilogServerImpl/VerilogServer.h"
#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/LSP/Transport.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

#define DEBUG_TYPE "circt-verilog-lsp-server"

using namespace llvm;
using namespace llvm::lsp;

//===----------------------------------------------------------------------===//
// LSPServer
//===----------------------------------------------------------------------===//

namespace {

using ChangeEvent = llvm::lsp::TextDocumentContentChangeEvent;
using Bucket = circt::lsp::DocChangeBucket<ChangeEvent>;

struct LSPServer {

  using Bucket =
      circt::lsp::DocChangeBucket<llvm::lsp::TextDocumentContentChangeEvent>;

  LSPServer(const circt::lsp::LSPServerOptions &options,
            circt::lsp::VerilogServer &server, JSONTransport &transport)
      : server(server), transport(transport),
        documentChangeRegistry(
            /*factory=*/
            [&, options]() {
              if (options.disableDebounce)
                return std::make_shared<Bucket>();
              return std::make_shared<Bucket>(true, options.debounceMinMs,
                                              options.debounceMaxMs);
            }) {}

  //===--------------------------------------------------------------------===//
  // Initialization
  //===--------------------------------------------------------------------===//

  void onInitialize(const InitializeParams &params,
                    Callback<json::Value> reply);
  void onInitialized(const InitializedParams &params);
  void onShutdown(const NoParams &params, Callback<std::nullptr_t> reply);

  //===--------------------------------------------------------------------===//
  // Document Change
  //===--------------------------------------------------------------------===//

  void onDocumentDidOpen(const DidOpenTextDocumentParams &params);
  void onDocumentDidClose(const DidCloseTextDocumentParams &params);
  void onDocumentDidChange(const DidChangeTextDocumentParams &params);

  //===--------------------------------------------------------------------===//
  // Definitions and References
  //===--------------------------------------------------------------------===//

  void onGoToDefinition(const TextDocumentPositionParams &params,
                        Callback<std::vector<Location>> reply);
  void onReference(const ReferenceParams &params,
                   Callback<std::vector<Location>> reply);

  //===--------------------------------------------------------------------===//
  // Fields
  //===--------------------------------------------------------------------===//

  circt::lsp::VerilogServer &server;
  JSONTransport &transport;

  /// An outgoing notification used to send diagnostics to the client when they
  /// are ready to be processed.
  OutgoingNotification<PublishDiagnosticsParams> publishDiagnostics;
  /// A thread-safe version of `publishDiagnostics`
  void sendDiagnostics(const PublishDiagnosticsParams &p) {
    std::lock_guard<std::mutex> lk(diagnosticsMutex);
    publishDiagnostics(p); // serialize the write
  }

  /// Used to indicate that the 'shutdown' request was received from the
  /// Language Server client.
  std::atomic<bool> shutdownRequestReceived{false};
  circt::lsp::BucketRegistry<Bucket> documentChangeRegistry;

private:
  /// A mutex to serialize access to publishing diagnostics
  std::mutex diagnosticsMutex;
};

} // namespace
//===----------------------------------------------------------------------===//
// Initialization
//===----------------------------------------------------------------------===//

void LSPServer::onInitialize(const InitializeParams &params,
                             Callback<json::Value> reply) {
  // Send a response with the capabilities of this server.
  json::Object serverCaps{
      {
          "textDocumentSync",
          llvm::json::Object{
              {"openClose", true},
              {"change", (int)TextDocumentSyncKind::Incremental},
              {"save", true},

          },

      },
      {"definitionProvider", true},
      {"referencesProvider", true},
  };

  json::Object result{
      {{"serverInfo", json::Object{{"name", "circt-verilog-lsp-server"},
                                   {"version", "0.0.1"}}},
       {"capabilities", std::move(serverCaps)}}};
  reply(std::move(result));
}
void LSPServer::onInitialized(const InitializedParams &) {}
void LSPServer::onShutdown(const NoParams &, Callback<std::nullptr_t> reply) {
  shutdownRequestReceived.store(true, std::memory_order_relaxed);
  documentChangeRegistry.cancelAll();
  reply(nullptr);
}

//===----------------------------------------------------------------------===//
// Document Change
//===----------------------------------------------------------------------===//

void LSPServer::onDocumentDidOpen(const DidOpenTextDocumentParams &params) {
  PublishDiagnosticsParams diagParams(params.textDocument.uri,
                                      params.textDocument.version);
  server.addDocument(params.textDocument.uri, params.textDocument.text,
                     params.textDocument.version, diagParams.diagnostics);

  // Publish any recorded diagnostics.
  sendDiagnostics(diagParams);
}

void LSPServer::onDocumentDidClose(const DidCloseTextDocumentParams &params) {
  std::optional<int64_t> version =
      server.removeDocument(params.textDocument.uri);
  if (!version)
    return;
  documentChangeRegistry.erase(params.textDocument.uri.file());

  // Empty out the diagnostics shown for this document. This will clear out
  // anything currently displayed by the client for this document (e.g. in the
  // "Problems" pane of VSCode).
  sendDiagnostics(PublishDiagnosticsParams(params.textDocument.uri, *version));
}

void LSPServer::onDocumentDidChange(const DidChangeTextDocumentParams &params) {
  auto fb = documentChangeRegistry.getOrCreate(params.textDocument.uri.file());
  // Buffer the raw change events (no text mutation yet)
  fb->enqueueChanges(params.contentChanges, params.textDocument.version);
  auto uri = params.textDocument.uri;

  // Schedule an update of the text file.
  fb->scheduleUpdate(
      [this,
       uri](const std::vector<llvm::lsp::TextDocumentContentChangeEvent> &batch,
            int64_t version) {
        PublishDiagnosticsParams diagParams(uri, version);
        server.updateDocument(diagParams.uri, batch, diagParams.version,
                              diagParams.diagnostics);
        sendDiagnostics(diagParams);
      });
}

//===----------------------------------------------------------------------===//
// Definitions and References
//===----------------------------------------------------------------------===//

void LSPServer::onGoToDefinition(const TextDocumentPositionParams &params,
                                 Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.getLocationsOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

void LSPServer::onReference(const ReferenceParams &params,
                            Callback<std::vector<Location>> reply) {
  std::vector<Location> locations;
  server.findReferencesOf(params.textDocument.uri, params.position, locations);
  reply(std::move(locations));
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult
circt::lsp::runVerilogLSPServer(const circt::lsp::LSPServerOptions &options,
                                VerilogServer &server,
                                JSONTransport &transport) {
  LSPServer lspServer(options, server, transport);
  MessageHandler messageHandler(transport);

  // Initialization
  messageHandler.method("initialize", &lspServer, &LSPServer::onInitialize);
  messageHandler.notification("initialized", &lspServer,
                              &LSPServer::onInitialized);
  messageHandler.method("shutdown", &lspServer, &LSPServer::onShutdown);

  // Document Changes
  messageHandler.notification("textDocument/didOpen", &lspServer,
                              &LSPServer::onDocumentDidOpen);
  messageHandler.notification("textDocument/didClose", &lspServer,
                              &LSPServer::onDocumentDidClose);

  messageHandler.notification("textDocument/didChange", &lspServer,
                              &LSPServer::onDocumentDidChange);
  // Definitions and References
  messageHandler.method("textDocument/definition", &lspServer,
                        &LSPServer::onGoToDefinition);
  messageHandler.method("textDocument/references", &lspServer,
                        &LSPServer::onReference);

  // Diagnostics
  lspServer.publishDiagnostics =
      messageHandler.outgoingNotification<PublishDiagnosticsParams>(
          "textDocument/publishDiagnostics");

  // Run the main loop of the transport.
  if (Error error = transport.run(messageHandler)) {
    Logger::error("Transport error: {0}", error);
    consumeError(std::move(error));
    return failure();
  }

  return success(lspServer.shutdownRequestReceived);
}
