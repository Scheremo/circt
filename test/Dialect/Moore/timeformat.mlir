// RUN: circt-opt %s | FileCheck %s

// CHECK: moore.builtin.timeformat -9, 2, " ns", 6
moore.builtin.timeformat -9, 2, " ns", 6

// Optional args omitted
// CHECK: moore.builtin.timeformat 0, 0
moore.builtin.timeformat 0, 0
