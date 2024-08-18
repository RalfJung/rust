//@ compile-flags: -Cno-prepopulate-passes
//@ needs-unwind
//@ min-llvm-version: 17.0.2
#![crate_type = "lib"]

// This test checks that drop calls in unwind landing pads
// get the `cold` attribute.

// CHECK-LABEL: @check_cold
// CHECK: {{(call|invoke) void .+}}drop_in_place{{.+}} [[ATTRIBUTES:#[0-9]+]]
// CHECK: attributes [[ATTRIBUTES]]
// CHECK-SAME: cold
#[no_mangle]
pub fn check_cold(f: fn(), x: Box<u32>) {
    // this may unwind
    f();
}
