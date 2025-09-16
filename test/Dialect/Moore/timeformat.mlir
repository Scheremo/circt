// RUN: circt-opt %s --moore-annotate-timeformat | FileCheck %s

moore.module @M() {
// With all params, first should have defaults
// CHECK: moore.builtin.timeformat(-9, 2, " ns", 6) {resolved_timeformat = #moore.timeformat<-15, -15, "", 20>}
moore.builtin.timeformat (-9, 2, " ns", 6)

// Optional params omitted
// CHECK: moore.builtin.timeformat {resolved_timeformat = #moore.timeformat<-9, 2, " ns", 6>}
moore.builtin.timeformat

// Also allow attributes on the op (for sanity), needs to have defaults again
// CHECK: moore.builtin.timeformat (1, -8, " ns", 17) {foo = 1 : i32, resolved_timeformat = #moore.timeformat<-15, -15, "", 20>}
moore.builtin.timeformat (1, -8, " ms", 17) {foo = 1 : i32}
}
