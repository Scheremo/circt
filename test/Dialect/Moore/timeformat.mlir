// RUN: circt-opt %s | FileCheck %s

// With all params
// CHECK: moore.builtin.timeformat -9, 2, " ns", 6
moore.builtin.timeformat -9, 2, " ns", 6

// Optional params omitted
// CHECK: moore.builtin.timeformat 0, 0
moore.builtin.timeformat 0, 0

// Also allow attributes on the op (for sanity)
// CHECK: moore.builtin.timeformat -9, 2, " ns", 6 {foo = 1 : i32}
moore.builtin.timeformat -9, 2, " ns", 6 {foo = 1 : i32}
