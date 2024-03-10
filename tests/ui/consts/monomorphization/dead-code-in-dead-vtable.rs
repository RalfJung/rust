//@revisions: opt no-opt
//@ build-fail
//@[opt] compile-flags: -O
//! Make sure we detect erroneous constants post-monomorphization even when they are unused. This is
//! crucial, people rely on it for soundness. (https://github.com/rust-lang/rust/issues/112090)

struct Fail<T>(T);
impl<T> Fail<T> {
    const C: () = panic!(); //~ERROR evaluation of `Fail::<i32>::C` failed
}

trait MyTrait {
    fn not_called(&self);
}

// This function is not actually called, but it is mentioned in a vtable in a function that is
// called. Make sure we still find this error.
impl<T> MyTrait for Vec<T> {
    fn not_called(&self) {
        let _ = Fail::<T>::C;
    }
}

#[inline(never)]
fn called<T>() {
    if false {
        let v: Vec<T> = Vec::new();
        let gen_vtable: &dyn MyTrait = &v;
    }
}

pub fn main() {
    called::<i32>();
}
